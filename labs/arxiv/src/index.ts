import Parser from 'rss-parser';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import "dotenv/config";
import { embed, embedMany, generateObject, cosineSimilarity } from "ai"
import { z } from "zod";
import { createOpenAI } from '@ai-sdk/openai';

import fs from "fs";

const google = createGoogleGenerativeAI({
    baseURL: process.env.AI_STUDIO_BASE_URL,
    apiKey: process.env.AI_STUDIO_API_KEY,
});

const openai = createOpenAI({
    baseURL: process.env.OPENAI_BASE_URL,
    apiKey: process.env.OPENAI_API_KEY,
})

const timeFilterMS = 1000 * 60 * 60 * 24 * 3; // 1.5 day

const parser = new Parser<{
    feedUrl: string,
    title: string,
    lastBuildDate: string,
    link: string,
}, {
    title: string,
    link: string,
    pubDate: string,
    author: string,
    summary: string,
    id: string,
    isoDate: string,
}>();

const allCategories = [
    'cs',
    'econ',
    'eess',
    'math',
    'astro-ph',
    'cond-mat',
    'gr-qc', // *無し
    'hep-ex', // *無し
    'hep-lat', // *無し
    'hep-ph',  // *無し
    'hep-th', // *無し
    'math-ph', // *無し
    'nlin',
    'nucl-ex', // *無し
    'nucl-th', // *無し
    'physics',
    'quant-ph', // *無し
    'q-bio',
    'q-fin',
    'stat'
]

const categoryQuery = "(cat:cs.* OR cat:econ.* OR cat:eess.* OR cat:math.* OR cat:astro-ph.* OR cat:cond-mat.* OR cat:gr-qc OR cat:hep-ex OR cat:hep-lat OR cat:hep-ph OR cat:hep-th OR cat:math-ph OR cat:nlin.* OR cat:nucl-ex OR cat:nucl-th OR cat:physics.* OR cat:quant-ph OR cat:q-bio.* OR cat:q-fin.* OR cat:stat.*)"



const getAllPaper = async () => {

    // papers.jsonがあるなら
    if (fs.existsSync("./papers.json")) {
        const papers = JSON.parse(fs.readFileSync("./papers.json").toString());
        return papers;
    }

    const papers: {
        author: string,
        title: string,
        link: string,
        summary: string,
        id: string,
        isoDate: string,
        pubDate: Date,
    }[] = [];


    // https://export.arxiv.org/api/query?search_query=au:del_maestro&submittedDate:[YYYYMMDDTTTT+TO+YYYYMMDDTTTT]

    // submittedDateはアメリカ東部時間ESTで指定する
    // 現在の時刻-timeFilterMSから現在の時刻までの論文を取得
    const now = new Date();
    const from = new Date(now.getTime() - timeFilterMS);
    from.setHours(from.getHours() - 5);
    now.setHours(now.getHours() - 5);
    const fromStr = `${from.getFullYear()}${String(from.getMonth() + 1).padStart(2, "0")}${String(from.getDate()).padStart(2, "0")}0000`;
    const nowStr = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}0000`;
    const submittedDate = `[${fromStr}+TO+${nowStr}]`;

    const search_query = `${encodeURIComponent(categoryQuery)}+AND+submittedDate:${submittedDate}`;

    while (true) {

        const url = `https://export.arxiv.org/api/query?search_query=${search_query}&max_results=500&start=${papers.length}`;

        console.log(url);
        const paper = await parser.parseURL(url);

        console.log(paper.items.length, papers.length + 1);

        if (paper.items.length === 0) {
            break;
        }

        for (const item of paper.items) {
            papers.push({
                author: item.author,
                title: item.title,
                link: item.link,
                summary: item.summary,
                id: item.id,
                isoDate: item.isoDate,
                pubDate: new Date(item.isoDate),
            });
        }

    }


    // ./papers.jsonに保存
    fs.writeFileSync("./papers.json", JSON.stringify(papers, null, 2));

    return papers;
}

const getGoogleEmbedding = async (text: string[]) => {
    const { embeddings } = await embedMany({
        model: google.textEmbeddingModel("text-embedding-004"),
        values: text
    })

    return embeddings.map((embedding, index) => {
        return {
            text: text[index],
            embedding: embedding
        }
    });
}

const userMetadata = {
    role: ['web engineer', 'llm engineer', 'hight shool student'],
    interest: ['machine learning', 'web development', 'cryptography', 'IoT', 'blockchain', 'social engineering'],
}

const main = async () => {
    const papers = await getAllPaper();

    // ロールと興味の埋め込みを取得
    const roleAndInterestEmbedding = await getGoogleEmbedding([...userMetadata.role, ...userMetadata.interest]);

    const userEmbedMetadata = {
        role: {
            raw: userMetadata.role,
            embedding: roleAndInterestEmbedding.filter((item) => userMetadata.role.includes(item.text)).map((item) => item.embedding)
        },
        interest: {
            raw: userMetadata.interest,
            embedding: roleAndInterestEmbedding.filter((item) => userMetadata.interest.includes(item.text)).map((item) => item.embedding)
        }
    }

    // const pickupIndex = [20, 42, 52, 224, 612, 432, 687, 3, 78, 33]
    // // const randomPapers = papers.sort(() => Math.random() - 0.5).slice(0, 10);
    // const randomPapers = papers.filter((_, index) => pickupIndex.includes(index));

    let inputToken = 0;
    let outputToken = 0;

    const embeddingRequests = new Set<string>();


    console.log("all papers", papers.length);

    const paperMetadataList: {
        author: string,
        title: string,
        link: string,
        summary: string,
        id: string,
        isoDate: string,
        pubDate: Date,
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
        finalScore: number,
        debug:{
            topicScore: number,
            targetScore: number,
            tagScore: number,
            topicSimilarities: number[],
            targetScores: number[],
            tagScores: number[],
        }
    }[] = [];

    await Promise.all(papers.map(async (paper) => {
        try {
            const result = await generateObject({
                model: google("gemini-2.0-flash", {
                    structuredOutputs: true,
                }),
                schema: z.object({
                    tags: z.array(z.string()).describe("Please tag words that are important in the paper and that categorize the paper."),
                    target: z.array(z.string()).describe("Please output the professions that will be most affected by this paper. E.g.) web engineer, infrastructure engineer, janitor"),
                    topic: z.string().describe("Please list the single most important topic in the paper."),
                    type: z.enum(["empirical", "theoretical", "literature", "experimental", "simulation"]).describe("Please indicate whether the paper is an empirical study/ theoretical study/ literature review/ experimental study/simulation."),
                }),
                prompt: `Please analyze the content based on the names and summaries of the following papers and generate JSON with meta-information.

# Instructions
Based on the information in the following papers, please summarize the tags, targets, keywords, topics, and types of papers. Please output the results in JSON format.
Each tag, target, and topic should be assigned a simple, short, precise word.

## Article Title.
${paper.title}

## Abstract of the paper
${paper.summary}`,
            });

            for (const tag of result.object.tags) {
                embeddingRequests.add(tag);
            }

            for (const target of result.object.target) {
                embeddingRequests.add(target);
            }

            embeddingRequests.add(result.object.topic);



            const paperMetadata = {
                ...paper,
                tags: {
                    raw: result.object.tags,
                },
                target: {
                    raw: result.object.target,
                },
                topic: {
                    raw: result.object.topic,
                },
                type: result.object.type,
            }

            paperMetadataList.push({
                ...paperMetadata
            });

            inputToken += result.usage.promptTokens;
            outputToken += result.usage.completionTokens;
            console.log(`Input tokens: ${inputToken}`);
            console.log(`Output tokens: ${outputToken}`);

            const progress = Math.round((paperMetadataList.length / papers.length) * 100);
            console.log(`LLM Progress: ${progress}% (${paperMetadataList.length}/${papers.length})`);

        } catch (e) {
            console.log(e);
        }
    }));

    // embeddingを取得する必要があるすべての文字列を一つの配列にまとめる
    const allEmbeddingRequests = Array.from(embeddingRequests);

    console.log("all embedding requests", allEmbeddingRequests.length);
    const embeddings: {
        text: string;
        embedding: number[];
    }[] = [];
    // Create chunks of 100 items
    const chunks = Array.from({ length: Math.ceil(allEmbeddingRequests.length / 100) }, (_, i) => 
        allEmbeddingRequests.slice(i * 100, (i + 1) * 100)
    );

    console.log("all chunks", chunks.length);

    // Process all chunks in parallel
    const embeddingResults = await Promise.all(
        chunks.map(async (chunk, index) => {
            const result = await getGoogleEmbedding(chunk);
            console.log(`Embedding Progress: ${Math.round(((index + 1) * 100 / chunks.length))}% (${(index + 1) * 100}/${allEmbeddingRequests.length})`);
            return result;
        })
    );

    // Flatten results into single array
    embeddings.push(...embeddingResults.flat());

    await Promise.all(paperMetadataList.map(async (paperMetadata) => {
        paperMetadata.tags.embedding = paperMetadata.tags.raw.map(tag => embeddings.find(embedding => embedding.text === tag)?.embedding || []);
        paperMetadata.target.embedding = paperMetadata.target.raw.map(target => embeddings.find(embedding => embedding.text === target)?.embedding || []);
        paperMetadata.topic.embedding = embeddings.find(embedding => embedding.text === paperMetadata.topic.raw)?.embedding || [];

        // 重みパラメータ（topicが最も重要）
        const TOPIC_WEIGHT = 4.0;
        const TARGET_WEIGHT = 2.0;
        const TAG_WEIGHT = 2.8;

        // 閾値（この値未満の類似度は無視する）
        const TOPIC_THRESHOLD = 0.7;
        const TARGET_THRESHOLD = 0.6;
        const TAG_THRESHOLD = 0.6;

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

        paperMetadata.finalScore = finalScore;
        paperMetadata.debug = {
            topicScore,
            targetScore,
            tagScore,
            topicSimilarities,
            targetScores,
            tagScores,
        }

        console.log(`Input tokens: ${inputToken}`);
        console.log(`Output tokens: ${outputToken}`);

        const progress = Math.round((paperMetadataList.length / papers.length) * 100);
        console.log(`Score Progress: ${progress}% (${paperMetadataList.length}/${papers.length})`);

    }));

    // 最終スコアでソート
    paperMetadataList.sort((a, b) => b.finalScore - a.finalScore);

    console.log("Recommended papers:");
    console.log(paperMetadataList.map(paper => `- ${paper.title} (Score: ${paper.finalScore.toFixed(3)})`));

    console.log("ALL Input tokens: ", inputToken);
    console.log("ALL Output tokens: ", outputToken);

    fs.writeFileSync("./result.json", JSON.stringify(paperMetadataList, null, 2));

}

main();

const test = async () => {
    // machine learningをembedding
    const machineLearningEmbedding = (await getGoogleEmbedding(["machine learning"]))[0].embedding;

    // web developmentをembedding
    const webDevelopmentEmbedding = (await getGoogleEmbedding(["web development"]))[0].embedding;

    // LLMをembedding
    const llmEmbedding = (await getGoogleEmbedding(["llm"]))[0].embedding;

    // hight shool studentをembedding
    const hightShoolStudentEmbedding = (await getGoogleEmbedding(["hight shool student"]))[0].embedding;

    // pattern matching をembedding
    const patternMatchingEmbedding = (await getGoogleEmbedding(["pattern matching"]))[0].embedding;

    // social engineering をembedding
    const socialEngineeringEmbedding = (await getGoogleEmbedding(["social engineering"]))[0].embedding;

    // cryptography をembedding
    const cryptographyEmbedding = (await getGoogleEmbedding(["cryptography"]))[0].embedding;

    // IoT をembedding
    const IoTEmbedding = (await getGoogleEmbedding(["IoT"]))[0].embedding;

    // machine learningとweb developmentの類似度をconsole.log
    console.log("machine learningとweb developmentの類似度", cosineSimilarity(machineLearningEmbedding, webDevelopmentEmbedding));
    console.log("machine learningとllmの類似度", cosineSimilarity(machineLearningEmbedding, llmEmbedding));
    console.log("machine learningとhight shool studentの類似度", cosineSimilarity(machineLearningEmbedding, hightShoolStudentEmbedding));
    console.log("machine learningとpattern matchingの類似度", cosineSimilarity(machineLearningEmbedding, patternMatchingEmbedding));
    console.log("machine learningとsocial engineeringの類似度", cosineSimilarity(machineLearningEmbedding, socialEngineeringEmbedding));
    console.log("machine learningとcryptographyの類似度", cosineSimilarity(machineLearningEmbedding, cryptographyEmbedding));
    console.log("machine learningとIoTの類似度", cosineSimilarity(machineLearningEmbedding, IoTEmbedding));

    // 類似度の順番でソートしてconsole.log
    const similarityList = [
        { name: "web development", embedding: webDevelopmentEmbedding },
        { name: "llm", embedding: llmEmbedding },
        { name: "hight shool student", embedding: hightShoolStudentEmbedding },
        { name: "pattern matching", embedding: patternMatchingEmbedding },
        { name: "social engineering", embedding: socialEngineeringEmbedding },
        { name: "cryptography", embedding: cryptographyEmbedding },
        { name: "IoT", embedding: IoTEmbedding },
    ].map((item) => ({
        name: item.name,
        similarity: cosineSimilarity(machineLearningEmbedding, item.embedding)
    })).sort((a, b) => b.similarity - a.similarity);

    console.log("machine learningと他の興味の類似度ランキング");

    similarityList.forEach((item, index) => {
        console.log(`${index + 1}. ${item.name}: ${item.similarity}`);
    });
}

// test();