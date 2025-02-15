import Parser from 'rss-parser';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import "dotenv/config";
import { generateObject } from "ai"
import z from "zod";
import { getEmbedding, EmbeddingIndex } from 'client-vector-search';
import search from "arXiv-api-ts";

const google = createGoogleGenerativeAI({
    baseURL: process.env.AI_STUDIO_BASE_URL,
    apiKey: process.env.AI_STUDIO_API_KEY,
});

const parser = new Parser<{
    feedUrl: string,
    paginationLinks: {
        self: string,
    },
    title: string,
    description: string,
    pubDate: string,
    managingEditor: string,
    link: string,
    language: string,
    lastBuildDate: string,
    docs: string,
    skipDays: {
        day: string[],
    }
}, {
    creator: string,
    rights: string,
    title: string,
    link: string,
    pubDate: string,
    "dc:creator": string,
    content: string,
    contentSnippet: string,
    guid: string,
    categories: string[],
    isoDate: string,
}>();

const allCategories = [
    'astro-ph',
    'cond-mat',
    'cs',
    'econ',
    'eess',
    'gr-qc',
    'hep-ex',
    'hep-lat',
    'hep-ph',
    'hep-th',
    'math',
    'math-ph',
    'nlin',
    'nucl-ex',
    'nucl-th',
    'physics',
    'q-bio',
    'q-fin',
    'quant-ph',
    'stat'
]

// 推奨スコアを算出するための補助関数：コサイン類似度の計算
const cosineSimilarity = (a: number[], b: number[]): number => {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (normA * normB);
};

const getAllPaper = async () => {

    const papers: {
        creator: string[],
        rights: string,
        title: string,
        link: string,
        pubDate: Date,
        content: string,
        guid: string,
        categories: string[]
    }[] = [];

    for (const category of allCategories) {
        const feed = await parser.parseURL(`https://rss.arxiv.org/rss/${category}`);

        for (const item of feed.items) {
            const creator = item.creator.split(', ');
            const rights = item.rights;
            const title = item.title;
            const link = item.link;
            const pubDate = new Date(item.pubDate);
            const content = item.content;
            const guid = item.guid;
            const categories = item.categories;



            papers.push({
                creator,
                rights,
                title,
                link,
                pubDate,
                content,
                guid,
                categories,
            });
        }
    }

    return papers;
}

const userMetadata = {
    role: ['web engineer', 'llm engineer', 'hight shool student'],
    interest: ['machine learning', 'web development', 'cryptography', 'IoT'],
}

const main = async () => {
    const papers = await getAllPaper();

    const userEmbedMetadata = {
        role: {
            raw: userMetadata.role,
            embedding: await Promise.all(userMetadata.role.map((role) => getEmbedding(role)))
        },
        interest: {
            raw: userMetadata.interest,
            embedding: await Promise.all(userMetadata.interest.map((interest) => getEmbedding(interest)))
        }
    }

    const pickupIndex = [20, 42, 52, 224, 612, 432, 687, 3, 78, 33]
    // const randomPapers = papers.sort(() => Math.random() - 0.5).slice(0, 10);
    const randomPapers = papers.filter((_, index) => pickupIndex.includes(index));


    console.log("all papers", randomPapers.length);

    const paperMetadataList: {
        creator: string[],
        rights: string,
        title: string,
        link: string,
        pubDate: Date,
        content: string,
        guid: string,
        categories: string[],
        tags: {
            raw: string[],
            embedding: number[][]
        },
        target: {
            raw: string[],
            embedding: number[][]
        },
        topic: {
            raw: string,
            embedding: number[]
        },
        type: string,
        finalScore: number
    }[] = [];

    for (const paper of randomPapers) {
        const result = await generateObject({
            model: google("gemini-2.0-flash", {
                structuredOutputs: false,
            }),
            schema: z.object({
                tags: z.array(z.string()),
                target: z.array(z.string()),
                topic: z.string(),
                type: z.enum(["empirical", "theoretical", "literature", "experimental", "simulation"]),
            }),
            prompt: `Please analyze the content based on the names and summaries of the following papers and generate JSON with meta-information.

# Instructions
Based on the information in the following papers, please summarize the tags, targets, keywords, topics, and types of papers.
The instructions for each type of information are as follows
- Tags: Please tag words that are important in the paper and that categorize the paper.
- Target: Please output the professions that will be most affected by this paper. E.g.) web engineer, infrastructure engineer, janitor
- Topic: Please list the single most important topic in the paper.
- Type of paper: Please indicate whether the paper is an empirical study/ theoretical study/ literature review/ experimental study/simulation.

## Article Title.
${paper.title}

## Abstract of the paper
${paper.content}`,
        });

        const paperMetadata = {
            ...paper,
            tags: {
                raw: result.object.tags,
                embedding: await Promise.all(result.object.tags.map((tag) => getEmbedding(tag)))
            },
            target: {
                raw: result.object.target,
                embedding: await Promise.all(result.object.target.map((target) => getEmbedding(target)))
            },
            topic: {
                raw: result.object.topic,
                embedding: await getEmbedding(result.object.topic)
            },
            type: result.object.type,
        }

        // 重みパラメータ（topicが最も重要）
        const TOPIC_WEIGHT = 3.0;
        const TARGET_WEIGHT = 2.5;
        const TAG_WEIGHT = 2.0;

        // 閾値（この値未満の類似度は無視する）
        const TOPIC_THRESHOLD = 0.8;
        const TARGET_THRESHOLD = 0.75;
        const TAG_THRESHOLD = 0.75;

        // ---------------------- レコメンデーションアルゴリズム ----------------------
        // 以下は、paperMetadataに含まれる埋め込み情報（tags, target, topic）と
        // userEmbedMetadataに格納されているユーザーの興味・役割の埋め込みとの類似度を
        // 計算し、重み付けして最終的なスコアを算出する処理例です。

        // 1. Topicの類似度
        //    論文のtopic埋め込みと、ユーザーの「interest」埋め込みそれぞれのコサイン類似度を計算し、最大値を採用します。
        const topicSimilarities = userEmbedMetadata.interest.embedding.map(userInterestEmbed =>
            cosineSimilarity(paperMetadata.topic.embedding, userInterestEmbed)
        );
        let topicScore = Math.max(...topicSimilarities);
        if (topicScore < TOPIC_THRESHOLD) topicScore = 0;

        // 2. Targetの類似度
        //    論文のtarget（職業候補）の各埋め込みと、ユーザーの「role」埋め込みとの類似度を計算し、
        //    各targetに対して最高の類似度を求めた後、全体として平均のスコアを採用します。
        const targetScores = paperMetadata.target.embedding.map(paperTargetEmbed => {
            const sims = userEmbedMetadata.role.embedding.map(userRoleEmbed =>
                cosineSimilarity(paperTargetEmbed, userRoleEmbed)
            );
            return Math.max(...sims);
        });
        let targetScore = targetScores.reduce((sum, score) => sum + score, 0) / targetScores.length;
        if (targetScore < TARGET_THRESHOLD) targetScore = 0;

        // 3. Tagsの類似度
        //    論文のtagsの各埋め込みと、ユーザーの「interest」埋め込みとの類似度を計算し、
        //    すべての組み合わせの中で平均を採用します。
        const tagScores = paperMetadata.tags.embedding.flatMap(tagEmbed =>
            userEmbedMetadata.interest.embedding.map(userInterestEmbed =>
                cosineSimilarity(tagEmbed, userInterestEmbed)
            )
        );
        let tagScore = tagScores.reduce((sum, score) => sum + score, 0) / tagScores.length;
        if (tagScore < TAG_THRESHOLD) tagScore = 0;

        // 4. 最終スコアの算出（各要素に重みをかけて合算）
        const finalScore = TOPIC_WEIGHT * topicScore + TARGET_WEIGHT * targetScore + TAG_WEIGHT * tagScore;

        console.log(`Paper: ${paperMetadata.title}`);
        console.log(`- Topic score: ${topicScore.toFixed(3)}`);
        console.log(`- Target score: ${targetScore.toFixed(3)}`);
        console.log(`- Tags score: ${tagScore.toFixed(3)}`);
        console.log(`=> Final recommendation score: ${finalScore.toFixed(3)}`);

        paperMetadataList.push({
            ...paperMetadata,
            finalScore
        });

        const progress = Math.round((paperMetadataList.length / randomPapers.length) * 100);
        console.log(`Progress: ${progress}% (${paperMetadataList.length}/${randomPapers.length})`);
    }

    // 最終スコアでソート
    paperMetadataList.sort((a, b) => b.finalScore - a.finalScore);

    console.log("Recommended papers:");
    console.log(paperMetadataList.map(paper => `- ${paper.title} (Score: ${paper.finalScore.toFixed(3)})`));
}

main();