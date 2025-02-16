import Parser from 'rss-parser';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import "dotenv/config";
import { embed, embedMany, generateObject, cosineSimilarity, generateText } from "ai"
import { z } from "zod";
import { createOpenAI } from '@ai-sdk/openai';
import prompts from 'prompts';

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



const getAllPaper = async (): Promise<{ author: string; title: string; link: string; summary: string; id: string; isoDate: string; pubDate: Date; }[]> => {

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
    role: ['blockchain engineer', 'web developer', 'data scientist'],
    interest: ['cryptography', 'zk', 'blockchain', "machine learning", "LLM", "healthcare", "social engineering", "IoT"],
    notInterest: ['biology', 'geology']
}

const getPaperMetadata = async (title: string, summary: string) => {
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
${title}

## Abstract of the paper
${summary}`,
    });

    return result.object;
}

const main = async () => {
    const papers = await getAllPaper();

    // ロールと興味の埋め込みを取得
    const roleAndInterestEmbedding = await getGoogleEmbedding([...userMetadata.role, ...userMetadata.interest, ...userMetadata.notInterest]);

    const userEmbedMetadata = {
        role: {
            raw: userMetadata.role,
            embedding: roleAndInterestEmbedding.filter((item) => userMetadata.role.includes(item.text)).map((item) => item.embedding)
        },
        interest: {
            raw: userMetadata.interest,
            embedding: roleAndInterestEmbedding.filter((item) => userMetadata.interest.includes(item.text)).map((item) => item.embedding)
        },
        notInterest: {
            raw: userMetadata.notInterest,
            embedding: roleAndInterestEmbedding.filter((item) => userMetadata.notInterest.includes(item.text)).map((item) => item.embedding)
        }
    }

    // const pickupIndex = [20, 42, 52, 224, 612, 432, 687, 3, 78, 33]
    // // const randomPapers = papers.sort(() => Math.random() - 0.5).slice(0, 10);
    // const randomPapers = papers.filter((_, index) => pickupIndex.includes(index));

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
            embedding?: number[][]
        },
        target: {
            raw: string[],
            embedding?: number[][]
        },
        topic: {
            raw: string,
            embedding?: number[]
        },
        type: string,
        finalScore?: number,
        debug?: {
            tagScore: number,
            targetScore: number,
            topicScore: number,
            tagScores: number[],
            targetScores: number[],
            topicSimilarities: number[],
            notInterestScore: number,
            notInterestScores: number[],
        }
    }[] = [];

    await Promise.all(papers.map(async (paper) => {
        try {
            const result = await getPaperMetadata(paper.title, paper.summary);

            for (const tag of result.tags) {
                embeddingRequests.add(tag);
            }

            for (const target of result.target) {
                embeddingRequests.add(target);
            }

            embeddingRequests.add(result.topic);



            const paperMetadata = {
                ...paper,
                tags: {
                    raw: result.tags,
                },
                target: {
                    raw: result.target,
                },
                topic: {
                    raw: result.topic,
                },
                type: result.type,
            }

            paperMetadataList.push({
                ...paperMetadata
            });

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
        const TAG_WEIGHT = 3.0;
        const NOT_INTEREST_WEIGHT = 1.0;

        // 閾値（この値未満の類似度は無視する）
        const TOPIC_THRESHOLD = 0.35;
        const TARGET_THRESHOLD = 0.35;
        const TAG_THRESHOLD = 0.35;

        // ---------------------- レコメンデーションアルゴリズム ----------------------
        // 以下は、paperMetadataに含まれる埋め込み情報（tags, target, topic）と
        // userEmbedMetadataに格納されているユーザーの興味・役割の埋め込みとの類似度を
        // 計算し、重み付けして最終的なスコアを算出する処理例です。

        // 1. Topicの類似度
        //    論文のtopic埋め込みと、ユーザーの「interest」埋め込みそれぞれのコサイン類似度を計算し、平均値を採用します。
        const topicSimilarities = userEmbedMetadata.interest.embedding.map(userInterestEmbed => {
            const similarity = cosineSimilarity(paperMetadata.topic.embedding, userInterestEmbed)
            if (similarity < TOPIC_THRESHOLD) return 0;
            return similarity;
        }
        );
        const topicScore = topicSimilarities.reduce((sum, score) => sum + score, 0) / topicSimilarities.length;

        // 2. Targetの類似度
        //    論文のtarget（職業候補）の各埋め込みと、ユーザーの「role」埋め込みとの類似度を計算し、
        //    各targetに対して最高の類似度を求めた後、全体として平均のスコアを採用します。
        const targetScores = paperMetadata.target.embedding.map(paperTargetEmbed => {
            const sims = userEmbedMetadata.role.embedding.map(userRoleEmbed => {
                const similarity = cosineSimilarity(paperTargetEmbed, userRoleEmbed)
                if (similarity < TARGET_THRESHOLD) return 0;
                return similarity;
            }
            );
            return Math.max(...sims);
        });
        const targetScore = targetScores.reduce((sum, score) => sum + score, 0) / targetScores.length;

        // 3. Tagsの類似度
        //    論文のtags（タグ）の各埋め込みと、ユーザーの「interest」埋め込みとの類似度を計算し、
        //    各interestに対して最高の類似度を求めた後、全体として平均のスコアを採用します。
        const tagScores = paperMetadata.tags.embedding.map(paperTagEmbed => {
            const sims = userEmbedMetadata.interest.embedding.map(userInterestEmbed => {
                const similarity = cosineSimilarity(paperTagEmbed, userInterestEmbed)
                if (similarity < TAG_THRESHOLD) return 0;
                return similarity;
            }
            );
            return Math.max(...sims);
        });
        const tagScore = tagScores.reduce((sum, score) => sum + score, 0) / tagScores.length;

        // 4. NotInterestの類似度
        //    論文のtags（タグ）の各埋め込みと、ユーザーの「notInterest」埋め込みとの類似度を計算し、
        //    各notInterestに対して最高の類似度を求めた後、全体として平均のスコアを採用します。
        const notInterestScores = paperMetadata.tags.embedding.map(paperTagEmbed => {
            const sims = userEmbedMetadata.notInterest.embedding.map(userNotInterestEmbed => {
                const similarity = cosineSimilarity(paperTagEmbed, userNotInterestEmbed)
                if (similarity < TAG_THRESHOLD) return 0;
                return similarity;
            }
            );
            return Math.max(...sims);
        });
        const notInterestScore = notInterestScores.reduce((sum, score) => sum + score, 0) / notInterestScores.length;

        // 4. 最終スコアの算出（各要素に重みをかけて合算）
        const finalScore = TOPIC_WEIGHT * topicScore + TARGET_WEIGHT * targetScore + TAG_WEIGHT * tagScore - NOT_INTEREST_WEIGHT * notInterestScore;

        console.log(`Paper: ${paperMetadata.title}`);
        console.log(`- Topic score: ${topicScore.toFixed(3)}`);
        console.log(`- Target score: ${targetScore.toFixed(3)}`);
        console.log(`- Tags score: ${tagScore.toFixed(3)}`);
        console.log(`- NotInterest score: ${notInterestScore.toFixed(3)}`);
        console.log(`=> Final recommendation score: ${finalScore.toFixed(3)}`);

        paperMetadata.finalScore = finalScore;
        paperMetadata.debug = {
            tagScore,
            targetScore,
            topicScore,
            tagScores,
            targetScores,
            topicSimilarities,
            notInterestScore,
            notInterestScores,
        }


        const progress = Math.round((paperMetadataList.length / papers.length) * 100);
        console.log(`Score Progress: ${progress}% (${paperMetadataList.length}/${papers.length})`);

    }));

    // 最終スコアでソート
    // biome-ignore lint/style/noNonNullAssertion: <explanation>
    paperMetadataList.sort((a, b) => b.finalScore! - a.finalScore!);

    console.log("Recommended papers:");
    console.log(paperMetadataList.map(paper => `- ${paper.title} (Score: ${paper.finalScore?.toFixed(3)})`));


    fs.writeFileSync("./result.json", JSON.stringify(paperMetadataList.map(v => {
        // embeddingを削除
        return {
            ...v,
            tags: v.tags.raw,
            target: v.target.raw,
            topic: v.topic.raw,
        }
    }), null, 2));

}

// main();

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

// ユーザーの興味を推測する

const googleTranslate = async (text: string) => {
    // geminiで翻訳
    const result = await generateText({
        model: google("gemini-2.0-flash"),
        system: "与えられた英語を日本語にわかりやすく訳して下さい。出力は訳した本文のみにしてください。それ以外を含めないでください。",
        prompt: text
    });

    return result.text;
}

function filterEmbeddingsByThreshold(group, compareGroup, threshold) {
    // group内の各論文(i番目)のタグ/ターゲットに対して
    // compareGroup内の「自分と同じ論文以外」のタグ/ターゲットと比較し
    // 最大類似度がthreshold以上なら残す
    return group.map((embeddingsArray, i) => {
        return embeddingsArray.filter((currentObj) => {
            let maxSimilarity = 0;

            // compareGroup をすべて回す。
            // 「自分と同じ論文 (i番目)」は除外したい場合は条件分岐
            // ただし、compareGroup と group が別物なら単純に全てを比較
            compareGroup.forEach((otherEmbeddingsArray, j) => {
                // もし group と compareGroup が同じなら、同じ論文 i を除外する
                if (group === compareGroup && i === j) {
                    return; // 同じ論文なのでスキップ
                }

                // 実際の embedding 同士を比較して最大値を更新
                for (const otherObj of otherEmbeddingsArray) {
                    const similarity = cosineSimilarity(currentObj.embedding, otherObj.embedding);
                    if (similarity > maxSimilarity) {
                        maxSimilarity = similarity;
                    }
                };
            });

            return maxSimilarity >= threshold;
        });
    });

}

async function selectPreferredPaper(papers: {
    translated: {
        title: string;
        summary: string;
    }
}[], index1: number, index2: number): Promise<number> {
    // 2つの論文を表示
    console.log(`${index1 + 1}.`, papers[index1].translated.title);
    console.log(papers[index1].translated.summary);
    console.log(`${index2 + 1}.`, papers[index2].translated.title);
    console.log(papers[index2].translated.summary);

    // 興味を尋ねる
    const { response } = await prompts({
        type: "select",
        name: "response",
        message: "どちらの論文が興味を引きましたか？",
        choices: [
            { title: papers[index1].translated.title, value: index1 },
            { title: papers[index2].translated.title, value: index2 }
        ]
    });

    return response;
}

const userInterest = async () => {
    const papers = await getAllPaper();

    // ランダムに20個の論文を選択
    const randomPapers = papers
        .sort(() => Math.random() - 0.5)
        .slice(0, 20);

    const papersTranslated = await Promise.all(randomPapers.map(async (paper) => {
        return {
            ...paper,
            translated: {
                title: await googleTranslate(paper.title),
                summary: await googleTranslate(paper.summary)
            }
        }
    }));

    // randomPapers[0]と[1]を表示
    console.log("========SELECT PAPER ========");
    const response1 = await selectPreferredPaper(papersTranslated, 0, 1);
    const response2 = await selectPreferredPaper(papersTranslated, 2, 3);
    const response3 = await selectPreferredPaper(papersTranslated, 4, 5);
    const response4 = await selectPreferredPaper(papersTranslated, 6, 7);
    const response5 = await selectPreferredPaper(papersTranslated, 8, 9);
    const response6 = await selectPreferredPaper(papersTranslated, 10, 11);
    const response7 = await selectPreferredPaper(papersTranslated, 12, 13);
    const response8 = await selectPreferredPaper(papersTranslated, 14, 15);
    const response9 = await selectPreferredPaper(papersTranslated, 16, 17);
    const response10 = await selectPreferredPaper(papersTranslated, 18, 19);


    // 興味を引いた論文を取得
    const interestedPapers = [response1, response2, response3, response4, response5, response6, response7, response8, response9, response10].map((index) => randomPapers[index]);
    // 興味を引かなかった論文を取得
    const notInterestedPapers = randomPapers.filter((paper) => !interestedPapers.includes(paper));

    // すべての論文にタグ、ターゲット、トピックを生成

    const interestedPaperMetadata = await Promise.all(interestedPapers.map(async (paper) => {
        const object = await getPaperMetadata(paper.title, paper.summary);
        return {
            ...paper,
            tags: {
                raw: object.tags
            },
            target: {
                raw: object.target
            },
            topic: {
                raw: object.topic
            },
        }
    }));

    const notInterestedPaperMetadata = await Promise.all(notInterestedPapers.map(async (paper) => {
        const object = await getPaperMetadata(paper.title, paper.summary);
        return {
            ...paper,
            tags: {
                raw: object.tags
            },
            target: {
                raw: object.target
            },
            topic: {
                raw: object.topic
            },
        }
    }));

    const interestedTags = interestedPaperMetadata.map((paper) => paper.tags.raw);
    const interestedTargets = interestedPaperMetadata.map((paper) => paper.target.raw);

    const notInterestedTags = notInterestedPaperMetadata.map((paper) => paper.tags.raw);
    const notInterestedTargets = notInterestedPaperMetadata.map((paper) => paper.target.raw);

    const notInterestedTagsFiltered = notInterestedTags.filter((tag) => !interestedTags.includes(tag));
    const notInterestedTargetsFiltered = notInterestedTargets.filter((target) => !interestedTargets.includes(target));

    // すべてのタグとターゲットをembedding
    const allTags = [...interestedTags, ...notInterestedTagsFiltered];
    const allTargets = [...interestedTargets, ...notInterestedTargetsFiltered];

    // Create chunks of 100 items to avoid too many requests
    const allTagsChunks = Array.from(
        { length: Math.ceil(allTags.flat().length / 100) },
        (_, i) => allTags.flat().slice(i * 100, (i + 1) * 100)
    );

    const allTargetsChunks = Array.from(
        { length: Math.ceil(allTargets.flat().length / 100) },
        (_, i) => allTargets.flat().slice(i * 100, (i + 1) * 100)
    );

    // Process chunks sequentially
    const allTagsEmbedding = (await Promise.all(
        allTagsChunks.map(async chunk => {
            return await getGoogleEmbedding(chunk);
        })
    )).flat();

    const allTargetsEmbedding = (await Promise.all(
        allTargetsChunks.map(async chunk => {
            return await getGoogleEmbedding(chunk);
        })
    )).flat();

    // interestedTags,notInterestedTagsFiltered,interestedTargets,notInterestedTargetsFilteredの値にそれぞれのvalueに対応するembeddingを追加していく
    const interestedTagsEmbedding = interestedTags.map((tags) => tags.map((tag) => {
        return {
            raw: tag,
            embedding: allTagsEmbedding.find((item) => item.text === tag)?.embedding || []
        }
    }));

    const interestedTargetsEmbedding = interestedTargets.map((targets) => targets.map((target) => {
        return {
            raw: target,
            embedding: allTargetsEmbedding.find((item) => item.text === target)?.embedding || []
        }
    }));

    const notInterestedTagsFilteredEmbedding = notInterestedTagsFiltered.map((tags) => tags.map((tag) => {
        return {
            raw: tag,
            embedding: allTagsEmbedding.find((item) => item.text === tag)?.embedding || []
        }
    }));

    const notInterestedTargetsFilteredEmbedding = notInterestedTargetsFiltered.map((targets) => targets.map((target) => {
        return {
            raw: target,
            embedding: allTargetsEmbedding.find((item) => item.text === target)?.embedding || []
        }
    }));

    const INTEREST_THRESHOLD = 0.75;
    const NOT_INTEREST_THRESHOLD = 0.5;

    // タグに対してフィルタをかける
    const filteredInterestedTagsEmbedding = filterEmbeddingsByThreshold(
        interestedTagsEmbedding,
        // 「自分以外の論文」と比較するなら、「notInterestedTagsFilteredEmbedding」と両方比較したいケースなどもある
        // シンプルに “すべて” と比較したいなら連結して渡す
        interestedTagsEmbedding,
        INTEREST_THRESHOLD
    );

    const filteredNotInterestedTagsFilteredEmbedding = filterEmbeddingsByThreshold(
        notInterestedTagsFilteredEmbedding,
        // 興味ありのもの + 興味なしの別のもの すべてと比較する場合
        notInterestedTagsFilteredEmbedding,
        NOT_INTEREST_THRESHOLD
    );

    // ターゲットに対してフィルタをかける
    const filteredInterestedTargetsEmbedding = filterEmbeddingsByThreshold(
        interestedTargetsEmbedding,
        interestedTargetsEmbedding,
        INTEREST_THRESHOLD
    );

    const filteredNotInterestedTargetsFilteredEmbedding = filterEmbeddingsByThreshold(
        notInterestedTargetsFilteredEmbedding,
        notInterestedTargetsFilteredEmbedding,
        NOT_INTEREST_THRESHOLD
    );

    // interestedTagsEmbeddingの中でそれぞれの論文の中で最も同じ論文内の類似度の平均が高いものを取得
    const interestedTagsMostSimilarEmbedding = [];
    for (let i = 0; i < interestedTagsEmbedding.length; i++) {
        const similarity: {
            raw: string,
            similarity: number,
            embedding: number[]
        }[] = [];
        for (let j = 0; j < interestedTagsEmbedding[i].length; j++) {
            let maxSimilarity = 0;
            for (let k = 0; k < interestedTagsEmbedding[i].length; k++) {
                if (j === k) continue;
                const currentSimilarity = cosineSimilarity(interestedTagsEmbedding[i][j].embedding, interestedTagsEmbedding[i][k].embedding);
                if (currentSimilarity > maxSimilarity) {
                    maxSimilarity = currentSimilarity;
                }
            }
            similarity.push({
                raw: interestedTagsEmbedding[i][j].raw,
                similarity: maxSimilarity,
                embedding: interestedTagsEmbedding[i][j].embedding
            });
        }

        similarity.sort((a, b) => b.similarity - a.similarity);
        interestedTagsMostSimilarEmbedding.push({
            raw: similarity[0].raw,
            embedding: similarity[0].embedding
        });
    }

    // interestedTargetsEmbeddingの中でそれぞれの論文の中で最も同じ論文内の類似度の平均が高いものを取得
    const interestedTargetsMostSimilarEmbedding = [];
    for (let i = 0; i < interestedTargetsEmbedding.length; i++) {
        const similarity: {
            raw: string,
            similarity: number,
            embedding: number[]
        }[] = [];
        for (let j = 0; j < interestedTargetsEmbedding[i].length; j++) {
            let maxSimilarity = 0;
            for (let k = 0; k < interestedTargetsEmbedding[i].length; k++) {
                if (j === k) continue;
                const currentSimilarity = cosineSimilarity(interestedTargetsEmbedding[i][j].embedding, interestedTargetsEmbedding[i][k].embedding);
                if (currentSimilarity > maxSimilarity) {
                    maxSimilarity = currentSimilarity;
                }
            }
            similarity.push({
                raw: interestedTargetsEmbedding[i][j].raw,
                similarity: maxSimilarity,
                embedding: interestedTargetsEmbedding[i][j].embedding
            });
        }

        similarity.sort((a, b) => b.similarity - a.similarity);
        interestedTargetsMostSimilarEmbedding.push({
            raw: similarity[0].raw,
            embedding: similarity[0].embedding
        });
    }

    // notInterestedTagsFilteredEmbeddingの中でそれぞれの論文の中で最も同じ論文内の類似度の平均が高いものを取得
    const notInterestedTagsMostSimilarEmbedding = [];
    for (let i = 0; i < notInterestedTagsFilteredEmbedding.length; i++) {
        const similarity: {
            raw: string,
            similarity: number,
            embedding: number[]
        }[] = [];
        for (let j = 0; j < notInterestedTagsFilteredEmbedding[i].length; j++) {
            let maxSimilarity = 0;
            for (let k = 0; k < notInterestedTagsFilteredEmbedding[i].length; k++) {
                if (j === k) continue;
                const currentSimilarity = cosineSimilarity(notInterestedTagsFilteredEmbedding[i][j].embedding, notInterestedTagsFilteredEmbedding[i][k].embedding);
                if (currentSimilarity > maxSimilarity) {
                    maxSimilarity = currentSimilarity;
                }
            }
            similarity.push({
                raw: notInterestedTagsFilteredEmbedding[i][j].raw,
                similarity: maxSimilarity,
                embedding: notInterestedTagsFilteredEmbedding[i][j].embedding
            });
        }

        similarity.sort((a, b) => b.similarity - a.similarity);
        notInterestedTagsMostSimilarEmbedding.push({
            raw: similarity[0].raw,
            embedding: similarity[0].embedding
        });
    }

    // notInterestedTargetsFilteredEmbeddingの中でそれぞれの論文の中で最も同じ論文内の類似度の平均が高いものを取得
    const notInterestedTargetsMostSimilarEmbedding = [];
    for (let i = 0; i < notInterestedTargetsFilteredEmbedding.length; i++) {
        const similarity: {
            raw: string,
            similarity: number,
            embedding: number[]
        }[] = [];
        for (let j = 0; j < notInterestedTargetsFilteredEmbedding[i].length; j++) {
            let maxSimilarity = 0;
            for (let k = 0; k < notInterestedTargetsFilteredEmbedding[i].length; k++) {
                if (j === k) continue;
                const currentSimilarity = cosineSimilarity(notInterestedTargetsFilteredEmbedding[i][j].embedding, notInterestedTargetsFilteredEmbedding[i][k].embedding);
                if (currentSimilarity > maxSimilarity) {
                    maxSimilarity = currentSimilarity;
                }
            }
            similarity.push({
                raw: notInterestedTargetsFilteredEmbedding[i][j].raw,
                similarity: maxSimilarity,
                embedding: notInterestedTargetsFilteredEmbedding[i][j].embedding
            });
        }

        similarity.sort((a, b) => b.similarity - a.similarity);
        notInterestedTargetsMostSimilarEmbedding.push({
            raw: similarity[0].raw,
            embedding: similarity[0].embedding
        });
    }



    const filteredInterestedTagsEmbeddingFlat = filteredInterestedTagsEmbedding.flat().concat(interestedTagsMostSimilarEmbedding);
    const filteredInterestedTargetsEmbeddingFlat = filteredInterestedTargetsEmbedding.flat().concat(interestedTargetsMostSimilarEmbedding);

    const filteredNotInterestedTagsFilteredEmbeddingFlat = filteredNotInterestedTagsFilteredEmbedding.flat().concat(notInterestedTagsMostSimilarEmbedding);
    const filteredNotInterestedTargetsFilteredEmbeddingFlat = filteredNotInterestedTargetsFilteredEmbedding.flat().concat(notInterestedTargetsMostSimilarEmbedding);

    // not interestedとinterestedでcos類似度が閾値を超えている場合、not interestedから削除
    const FilterTHRESHOLD = 0.65;
    const filteredNotInterestedTagsFilteredEmbeddingFiltered = filteredNotInterestedTagsFilteredEmbeddingFlat.filter((tag) => {
        for (const interestedTag of filteredInterestedTagsEmbeddingFlat) {
            if (cosineSimilarity(tag.embedding, interestedTag.embedding) > FilterTHRESHOLD) {
                return false;
            }
        }
        return true;
    });

    const filteredNotInterestedTargetsFilteredEmbeddingFiltered = filteredNotInterestedTargetsFilteredEmbeddingFlat.filter((target) => {
        for (const interestedTarget of filteredInterestedTargetsEmbeddingFlat) {
            if (cosineSimilarity(target.embedding, interestedTarget.embedding) > FilterTHRESHOLD) {
                return false;
            }
        }
        return true;
    });

    // flat
    const filteredNotInterestedTagsFilteredEmbeddingFilteredFlat = filteredNotInterestedTagsFilteredEmbeddingFiltered.map((item) => item.raw);
    const filteredNotInterestedTargetsFilteredEmbeddingFilteredFlat = filteredNotInterestedTargetsFilteredEmbeddingFiltered.map((item) => item.raw);

    console.log("Interested Tags");
    console.log([...new Set(filteredInterestedTagsEmbeddingFlat.map((tag) => tag.raw))]);

    console.log("Interested Targets");
    console.log([...new Set(filteredInterestedTargetsEmbeddingFlat.map((target) => target.raw))]);

    console.log("Not Interested Tags");
    console.log([...new Set(filteredNotInterestedTagsFilteredEmbeddingFilteredFlat)]);

    console.log("Not Interested Targets");
    console.log([...new Set(filteredNotInterestedTargetsFilteredEmbeddingFilteredFlat)]);
}

userInterest();