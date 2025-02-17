import { createGoogleGenerativeAI, type GoogleGenerativeAIProvider } from "@ai-sdk/google";
import Parser from 'rss-parser';
import fs from 'node:fs';
import { cosineSimilarity, embed, embedMany, generateObject, type Schema } from "ai";
import { z } from "zod";

const config: {
    threshold: {
        topic: number,
        target: number,
        tag: number,
        notInterest: number
    },
    weight: {
        topic: number,
        target: number,
        tag: number,
    },
    language: string,
    timeFilterMS: number,
    queryCategory: string
} = {
    threshold: {
        topic: 4,
        target: 2,
        tag: 3,
        notInterest: 1
    },
    weight: {
        topic: 0.35,
        target: 0.35,
        tag: 0.35
    },
    language: "ja",
    timeFilterMS: 1000 * 60 * 60 * 24 * 1, // 1 day
    queryCategory: "cat:cs.* OR cat:econ.* OR cat:eess.* OR cat:math.* OR cat:astro-ph.* OR cat:cond-mat.* OR cat:gr-qc OR cat:hep-ex OR cat:hep-lat OR cat:hep-ph OR cat:hep-th OR cat:math-ph OR cat:nlin.* OR cat:nucl-ex OR cat:nucl-th OR cat:physics.* OR cat:quant-ph OR cat:q-bio.* OR cat:q-fin.* OR cat:stat.*"
}


type Paper = {
    author: string,
    title: string,
    link: string,
    summary: string,
    id: string,
    isoDate: string,
    pubDate: Date,
}

type PaperMetadata = {
    topic: string,
    target: string[],
    tags: string[],
    type: string,
} & Paper

type PaperMetadataEmbedding = (Omit<PaperMetadata, "topic" | "target" | "tags"> & {
    topic: {
        embedding: number[],
        value: string,
    },
    target: {
        embedding: number[],
        value: string,
    }[],
    tags: {
        embedding: number[],
        value: string,
    }[],
})

type PaperMetadataScore = {
    topic: number,
    target: number,
    tag: number,
    notInterestTarget: number,
    notInterestTag: number,
}

type PaperMetadataWithScore = PaperMetadataEmbedding & {
    scores: PaperMetadataScore
}

const getArxivPapers = async (
    query: string,
    timeFilterMS: number,
) => {
    const papers: Paper[] = [];

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

    const now = new Date();
    const from = new Date(now.getTime() - timeFilterMS);
    from.setHours(from.getHours() - 5);
    now.setHours(now.getHours() - 5);
    const fromStr = `${from.getFullYear()}${String(from.getMonth() + 1).padStart(2, "0")}${String(from.getDate()).padStart(2, "0")}0000`;
    const nowStr = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}0000`;
    const submittedDate = `[${fromStr}+TO+${nowStr}]`;

    const search_query = `${encodeURIComponent(`(${query})`)}+AND+submittedDate:${submittedDate}`;

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

    return papers;
}

const getArxivPapersWithCache = async (
    query: string,
    timeFilterMS: number
) => {
    if (!fs.existsSync("./cache")) {
        fs.mkdirSync("./cache");
    }

    const now = new Date();
    const nowStr = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}`;

    const cachePath = `./cache/${nowStr}.json`;

    if (fs.existsSync(cachePath)) {
        const data = fs.readFileSync(cachePath, "utf-8");
        return JSON.parse(data);
    }

    const papers = await getArxivPapers(query, timeFilterMS);

    fs.writeFileSync(cachePath, JSON.stringify(papers));

    return papers;
}

class AITools {
    private google: GoogleGenerativeAIProvider;
    constructor(
        apiKey: string,
        baseURL?: string,
    ) {
        this.google = createGoogleGenerativeAI({
            baseURL,
            apiKey,
        });
    }

    public async getMultiEmbedding(text: string[]) {
        const CHUNK_SIZE = 150;
        const chunks: number[][] = [];

        for (let i = 0; i < text.length; i += CHUNK_SIZE) {
            const chunk = text.slice(i, i + CHUNK_SIZE);
            const { embeddings } = await embedMany({
                model: this.google.textEmbeddingModel("text-embedding-004"),
                values: chunk
            });
            chunks.push(...embeddings);
        }

        return chunks;
    }

    public async getEmbedding(text: string) {
        const { embedding } = await embed({
            model: this.google.textEmbeddingModel("text-embedding-004"),
            value: text
        })

        return embedding;
    }

    public async genObject<OBJECT>(
        prompt: string,
        // biome-ignore lint/suspicious/noExplicitAny: <explanation>
        schema: z.Schema<OBJECT, z.ZodTypeDef, any> | Schema<OBJECT>,
        system?: string,
    ): Promise<(null | OBJECT)> {

        try {
            const result = await generateObject({
                model: this.google("gemini-2.0-flash", {
                    structuredOutputs: true,
                }),
                schema,
                prompt,
                system,
            });

            return result.object;
        } catch (e) {
            console.error(e);
            return null;
        }
    }
}

type UserMetadata = {
    interest: {
        target: string[],
        tags: string[],
    },
    notInterest: {
        target: string[],
        tags: string[],
    }
}

type UserMetadataEmbedding = {
    interest: {
        target: {
            embedding: number[],
            value: string,
        }[],
        tags: {
            embedding: number[],
            value: string,
        }[],
    },
    notInterest: {
        target: {
            embedding: number[],
            value: string,
        }[],
        tags: {
            embedding: number[],
            value: string,
        }[],
    }
}

const getUserMetadataEmbedding = async (
    aiTools: AITools,
    metadata: UserMetadata
): Promise<UserMetadataEmbedding> => {
    // metadataの要素をそれぞれembeddingし、UserMetadataEmbeddingの型を返す。ただし実際のembeddingの呼び出しはgetMultiEmbeddingを1回だけ呼び出すこと。
    const target = metadata.interest.target.concat(metadata.notInterest.target);
    const tags = metadata.interest.tags.concat(metadata.notInterest.tags);

    const embeddings = await aiTools.getMultiEmbedding([...target, ...tags]);

    const userMetadataEmbedding: UserMetadataEmbedding = {
        interest: {
            target: target.map((value, index) => ({
                embedding: embeddings[index],
                value,
            })),
            tags: tags.map((value, index) => ({
                embedding: embeddings[index + target.length],
                value,
            }))
        },
        notInterest: {
            target: target.map((value, index) => ({
                embedding: embeddings[index],
                value,
            })),
            tags: tags.map((value, index) => ({
                embedding: embeddings[index + target.length],
                value,
            }))
        }
    }

    return userMetadataEmbedding;
}

const getPaperMetadata = async (
    aiTools: AITools,
    paper: Paper,
): Promise<PaperMetadata | null> => {

    const title = paper.title;
    const summary = paper.summary;

    const metadata = await aiTools.genObject(
        `Please analyze the content based on the names and summaries of the following papers and generate JSON with meta-information.

# Instructions
Based on the information in the following papers, please summarize the tags, targets, keywords, topics, and types of papers. Please output the results in JSON format.
Each tag, target, and topic should be assigned a simple, short, precise word.

## Article Title.
${title}

## Abstract of the paper
${summary}`,
        z.object({
            tags: z.array(z.string()).describe("Please tag words that are important in the paper and that categorize the paper."),
            target: z.array(z.string()).describe("Please output the professions that will be most affected by this paper. E.g.) web engineer, infrastructure engineer, janitor"),
            topic: z.string().describe("Please list the single most important topic in the paper."),
            type: z.enum(["empirical", "theoretical", "literature", "experimental", "simulation"]).describe("Please indicate whether the paper is an empirical study/ theoretical study/ literature review/ experimental study/simulation."),
        })
    );

    if (metadata === null) {
        return null;
    }

    return {
        ...paper,
        ...metadata,
    }
}

const getPapersMetadata = async (
    aiTools: AITools,
    papers: Paper[]
): Promise<PaperMetadata[]> => {
    const paperMetadata: PaperMetadata[] = [];

    for (const paper of papers) {
        const metadata = await getPaperMetadata(aiTools, paper);
        if (metadata !== null) {
            paperMetadata.push(metadata);
        }
    }

    return paperMetadata;
}

// 1対多のembeddingからcos類似度の最大値を取得する
const getMaxCosSimilarity = (embedding: number[], embeddings: number[][], threshold = 1) => {
    let max = -1;
    for (const target of embeddings) {
        const similarity = cosineSimilarity(embedding, target);
        if (similarity > max && similarity > threshold) {
            max = similarity;
        }
    }

    return max;
}

const getMaxCosSimilarityMany = (embeddings: number[][], targets: number[][], threshold = 1) => {
    let max = -1;
    for (const target of targets) {
        const similarity = getMaxCosSimilarity(target, embeddings);
        if (similarity > max && similarity > threshold) {
            max = similarity;
        }
    }

    return max;
}

const getAverageCosSimilarityMany = (embeddings: number[][], targets: number[][], threshold = 1) => {
    let sum = 0;
    for (const target of targets) {
        const similarity = getMaxCosSimilarity(target, embeddings);
        if (similarity > threshold) {
            sum += similarity;
        }
    }

    return sum / targets.length;
}

const getPapersMetadataEmbedding = async (
    aiTools: AITools,
    paperMetadata: PaperMetadata[]
): Promise<PaperMetadataEmbedding[]> => {
    const paperMetadataEmbedding: PaperMetadataEmbedding[] = [];

    const target = paperMetadata.map(paper => paper.target);
    const tags = paperMetadata.map(paper => paper.tags);
    const topics = paperMetadata.map(paper => paper.topic);

    const embeddings = await aiTools.getMultiEmbedding([...target.flat(), ...tags.flat(), ...topics]);

    for (const [index, paper] of paperMetadata.entries()) {
        const targetEmbeddings = paper.target.map((value, index) => ({
            embedding: embeddings[index],
            value,
        }));
        const tagEmbeddings = paper.tags.map((value, index) => ({
            embedding: embeddings[index + target.length],
            value,
        }));
        const topicEmbedding = {
            embedding: embeddings[index + target.length + tags.length],
            value: paper.topic,
        }

        paperMetadataEmbedding.push({
            ...paper,
            topic: topicEmbedding,
            target: targetEmbeddings,
            tags: tagEmbeddings,
        });
    }

    return paperMetadataEmbedding;
}

// ユーザーの好みに基づいて論文をスコアリングする
const scorePapers = async (
    aiTools: AITools,
    userMetadata: UserMetadata,
    papers: Paper[]
) => {
    const papersMetadata = await getPapersMetadata(aiTools, papers);
    const papersMetadataEmbedding = await getPapersMetadataEmbedding(aiTools, papersMetadata);
    const userMetadataEmbedding = await getUserMetadataEmbedding(aiTools, userMetadata);

    const papersMetadataWithScore: PaperMetadataWithScore[] = [];

    for (const paper of papersMetadataEmbedding) {
        const similarity = {
            interest: {
                tag: getAverageCosSimilarityMany(paper.tags.map(tag => tag.embedding), userMetadataEmbedding.interest.tags.map(tag => tag.embedding), config.threshold.tag),
                target: getMaxCosSimilarityMany(paper.target.map(target => target.embedding), userMetadataEmbedding.interest.target.map(target => target.embedding), config.threshold.target),
            },
            notInterest: {
                tag: getAverageCosSimilarityMany(paper.tags.map(tag => tag.embedding), userMetadataEmbedding.notInterest.tags.map(tag => tag.embedding), config.threshold.tag),
                target: getMaxCosSimilarityMany(paper.target.map(target => target.embedding), userMetadataEmbedding.notInterest.target.map(target => target.embedding), config.threshold.target),
            }
        }

        const finalScore = Object.values({
            topic: similarity.interest.target,
            target: similarity.interest.target,
            tag: similarity.interest.tag,
            notInterest: similarity.notInterest.target + similarity.notInterest.tag,
        }).reduce((acc, cur) => acc + cur, 0);

        papersMetadataWithScore.push({
            ...paper,
            scores: {
                topic: similarity.interest.target,
                target: similarity.interest.target,
                tag: similarity.interest.tag,
                notInterestTarget: -similarity.notInterest.target,
                notInterestTag: -similarity.notInterest.tag,
            }
        });
    }

    return papersMetadataWithScore;
}

const sortPapers = (papers: PaperMetadataWithScore[]) => {
    return papers.sort((a, b) => {
        const scoreA = Object.values(a.scores).reduce((acc, cur) => acc + cur, 0);
        const scoreB = Object.values(b.scores).reduce((acc, cur) => acc + cur, 0);

        return scoreB - scoreA;
    });
}

const getRecommendedPapers = async (
    aiTools: AITools,
    userMetadata: UserMetadata,
    queryCategory: string,
    timeFilterMS: number
) => {
    const papers = await getArxivPapersWithCache(queryCategory, timeFilterMS);
    const scoredPapers = await scorePapers(aiTools, userMetadata, papers);
    const sortedPapers = sortPapers(scoredPapers);

    return sortedPapers;
}

const main = async () => {
    const userMetadata = {
        interest: {
            target: ['blockchain engineer', 'web developer', 'data scientist'],
            tags: ['cryptography', 'zk', 'blockchain', "machine learning", "LLM", "healthcare", "social engineering", "IoT"],
        },
        notInterest: {
            target: ['biology', 'geology'],
            tags: ['biology scientist', 'geology scientist'],
        }
    }

    const aiTools = new AITools(process.env.AI_STUDIO_API_KEY || "", process.env.AI_STUDIO_BASE_URL);

    const papers = await getRecommendedPapers(aiTools, userMetadata, config.queryCategory, config.timeFilterMS);

    console.table(papers.slice(0, 10).map(paper => ({
        title: paper.title,
        link: paper.link,
        score: Object.values(paper.scores).reduce((acc, cur) => acc + cur, 0),
    })));
}

main();