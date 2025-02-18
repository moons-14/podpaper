import { createGoogleGenerativeAI, type GoogleGenerativeAIProvider } from "@ai-sdk/google";
import Parser from 'rss-parser';
import fs from 'node:fs';
import { cosineSimilarity, embed, embedMany, generateObject, type Schema } from "ai";
import { z } from "zod";
import "dotenv/config";

const config: {
    threshold: {
        interest: number,
        notInterest: number,
    },
    weight: {
        topic: number,
        target: number,
        tag: number,
        notInterest: number
    },
    language: string,
    timeFilterMS: number,
    sigmoid_k: number,
    queryCategory: string
} = {
    threshold: {
        interest: 0.35,
        notInterest: 0.6,
    },
    weight: {
        topic: 4,
        target: 2,
        tag: 3,
        notInterest: 2.0
    },
    language: "ja",
    timeFilterMS: 1000 * 60 * 60 * 24 * 4, // 4 day
    sigmoid_k: 0.5,
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
    final: number,
}

type PaperMetadataWithScore = PaperMetadataEmbedding & {
    scores: PaperMetadataScore
}

const sigmoid = (x: number, k = 1): number => 1 / (1 + Math.exp(-k * x));

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

        const paper = await parser.parseURL(url);

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

    const cachePath = `./cache/${nowStr}-${timeFilterMS}.json`;

    if (fs.existsSync(cachePath)) {
        const data = fs.readFileSync(cachePath, "utf-8");
        const papers = JSON.parse(data);

        console.debug("cache hit");
        console.debug("cache path", cachePath);
        console.debug("papers length", papers.length);

        return papers;
    }

    console.debug("cache miss");
    console.debug("getting papers....");

    const papers = await getArxivPapers(query, timeFilterMS);

    console.debug("got papers");
    console.debug("papers length", papers.length);

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

    public async getMultiEmbedding(textList: string[]) {
        const CHUNK_SIZE = 100;
        const chunks: {
            embedding: number[],
            value: string,
        }[] = [];

        const chunkArrays: string[][] = [];
        for (let i = 0; i < textList.length; i += CHUNK_SIZE) {
            chunkArrays.push(textList.slice(i, i + CHUNK_SIZE));
        }

        const embeddingPromises = chunkArrays.map(chunk =>
            embedMany({
                model: this.google.textEmbeddingModel("text-embedding-004"),
                values: chunk
            })
        );

        const results = await Promise.all(embeddingPromises);

        for (let i = 0; i < results.length; i++) {
            const { embeddings } = results[i];
            chunks.push(...embeddings.map((embedding, index) => ({
                embedding,
                value: chunkArrays[i][index],
            })));
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
    console.debug("getting user metadata embedding...");

    const allEmbeddingText = [...new Set([...metadata.interest.target, ...metadata.notInterest.target, ...metadata.interest.tags, ...metadata.notInterest.tags])];
    const embeddings = await aiTools.getMultiEmbedding(allEmbeddingText);

    const userMetadataEmbedding: UserMetadataEmbedding = {
        interest: {
            target: metadata.interest.target.map((value) => embeddings.find(embedding => embedding.value === value)).filter(v => !!v),
            tags: metadata.interest.tags.map((value) => embeddings.find(embedding => embedding.value === value)).filter(v => !!v),
        },
        notInterest: {
            target: metadata.notInterest.target.map((value) => embeddings.find(embedding => embedding.value === value)).filter(v => !!v),
            tags: metadata.notInterest.tags.map((value) => embeddings.find(embedding => embedding.value === value)).filter(v => !!v),
        }
    }

    console.debug("got user metadata embedding");
    return userMetadataEmbedding;
}

const getPaperMetadata = async (
    aiTools: AITools,
    paper: Paper,
    userMetadata: UserMetadata
): Promise<PaperMetadata | null> => {

    const pickupUserMetadata = {
        interest: {
            tags: userMetadata.interest.tags.sort(() => Math.random() - 0.5).slice(0, 10).map(tag => tag.toLowerCase().trim()),
            target: userMetadata.interest.target.sort(() => Math.random() - 0.5).slice(0, 10).map(target => target.toLowerCase().trim())
        },
        notInterest: {
            tags: userMetadata.notInterest.tags.sort(() => Math.random() - 0.5).slice(0, 10).map(tag => tag.toLowerCase().trim()),
            target: userMetadata.notInterest.target.sort(() => Math.random() - 0.5).slice(0, 10).map(target => target.toLowerCase().trim())
        }
    }

    const title = paper.title;
    const summary = paper.summary;

    const metadata = await aiTools.genObject(
        `Please analyze the content based on the names and summaries of the following papers and generate JSON with meta-information.

# Instructions
Based on the information in the following papers, please summarize the tags, targets, keywords, topics, and types of papers. Please output the results in JSON format.
please do not use words that are too common or words that can be abbreviated to obscure the context. Please tag them reliably and clearly.
However, please see the following example of a user's tag and be aware of the relative nature of the output to that tag.

## User's tag
- InterestedTarget: ${pickupUserMetadata.interest.target.join(", ")}
- InterestedTags: ${pickupUserMetadata.interest.tags.join(", ")}
- notInterestedTarget: ${pickupUserMetadata.notInterest.target.join(", ")}
- notInterestedTags: ${pickupUserMetadata.notInterest.tags.join(", ")}

## Article Title.
${title}

## Abstract of the paper
${summary.replace(/\n/g, " ")}`,
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
    papers: Paper[],
    userMetadata: UserMetadata
): Promise<PaperMetadata[]> => {
    console.debug("getting papers metadata...");

    const metadataPromises = papers.map(paper => getPaperMetadata(aiTools, paper, userMetadata));
    const results = await Promise.all(metadataPromises);

    const paperMetadata = results.filter(metadata => metadata !== null) as PaperMetadata[];

    console.debug("got papers metadata");
    console.debug("papers metadata length", paperMetadata.length);

    return paperMetadata;
};


// 1対多のembeddingからcos類似度の最大値を取得する
const getMaxCosSimilarity = (embedding: number[], embeddings: number[][], threshold = 0) => {
    let max = -1;
    for (const target of embeddings) {
        const similarity = cosineSimilarity(embedding, target);
        if (similarity > max && similarity > threshold) {
            max = similarity;
        }
    }

    return max;
}

const getMaxCosSimilarityMany = (embeddings: number[][], targets: number[][], threshold = 0) => {
    let max = -1;
    for (const target of targets) {
        const similarity = getMaxCosSimilarity(target, embeddings);
        if (similarity > max && similarity > threshold) {
            max = similarity;
        }
    }

    return max;
}

const getAverageCosSimilarityMany = (embeddings: number[][], targets: number[][], threshold = 0) => {
    let sum = 0;
    for (const target of targets) {
        const similarity = getMaxCosSimilarity(target, embeddings);
        if (similarity > threshold) {
            sum += similarity;
        }
    }

    return sum / targets.length;
}

const getCombinedCosSimilarity = (
    embeddings: number[][],
    targets: number[][],
    threshold = 0,
    alpha = 0.6
): number => {
    let maxSim = -1;
    let sumSim = 0;
    let count = 0;

    for (const target of targets) {
        // Compute the maximum similarity for this target against all embeddings.
        let localMax = -1;
        for (const embedding of embeddings) {
            const sim = cosineSimilarity(embedding, target);
            if (sim > localMax) {
                localMax = sim;
            }
        }
        // Only consider values above the threshold.
        if (localMax > threshold) {
            sumSim += localMax;
            count++;
            if (localMax > maxSim) {
                maxSim = localMax;
            }
        }
    }
    if (count === 0) {
        return 0;
    }
    const avgSim = sumSim / count;
    // Combine max and average similarities.
    return alpha * maxSim + (1 - alpha) * avgSim;
};

const getPapersMetadataEmbedding = async (
    aiTools: AITools,
    paperMetadata: PaperMetadata[]
): Promise<PaperMetadataEmbedding[]> => {
    console.debug("getting papers metadata embedding...");
    const paperMetadataEmbedding: PaperMetadataEmbedding[] = [];

    const target = paperMetadata.map(paper => paper.target);
    const tags = paperMetadata.map(paper => paper.tags);
    const topics = paperMetadata.map(paper => paper.topic);

    const allEmbeddingText = [...new Set([...target.flat(), ...tags.flat(), ...topics])];
    const embeddings = await aiTools.getMultiEmbedding(allEmbeddingText);

    await Promise.all(paperMetadata.map(async (paper) => {
        const targetEmbeddings = paper.target.map(target => embeddings.find(embedding => embedding.value === target)).filter(v => !!v);
        const tagEmbeddings = paper.tags.map(tag => embeddings.find(embedding => embedding.value === tag)).filter(v => !!v);
        const topicEmbedding = embeddings.find(embedding => embedding.value === paper.topic) || { embedding: [], value: "" };

        paperMetadataEmbedding.push({
            ...paper,
            topic: topicEmbedding,
            target: targetEmbeddings,
            tags: tagEmbeddings,
        });
    }));

    console.debug("got papers metadata embedding");

    return paperMetadataEmbedding;
}

// ユーザーの好みに基づいて論文をスコアリングする
const scorePapers = async (
    aiTools: AITools,
    userMetadata: UserMetadata,
    papers: Paper[]
) => {
    console.debug("scoring papers...");

    const papersMetadata = await getPapersMetadata(aiTools, papers, userMetadata);
    const papersMetadataEmbedding = await getPapersMetadataEmbedding(aiTools, papersMetadata);
    const userMetadataEmbedding = await getUserMetadataEmbedding(aiTools, userMetadata);

    const scoringPromises = papersMetadataEmbedding.map(async (paper) => {
        const similarity = {
            topic: getCombinedCosSimilarity(
                [paper.topic.embedding],
                userMetadataEmbedding.interest.tags.map(tag => tag.embedding),
                config.threshold.interest
            ),
            target: getCombinedCosSimilarity(
                paper.target.map(target => target.embedding),
                userMetadataEmbedding.interest.target.map(target => target.embedding),
                config.threshold.interest
            ),
            tag: getCombinedCosSimilarity(
                paper.tags.map(tag => tag.embedding),
                userMetadataEmbedding.interest.tags.map(tag => tag.embedding),
                config.threshold.interest
            ),
            notInterestedTarget: getCombinedCosSimilarity(
                paper.target.map(target => target.embedding),
                userMetadataEmbedding.notInterest.target.map(target => target.embedding),
                config.threshold.notInterest
            ),
            notInterestedTag: getCombinedCosSimilarity(
                paper.tags.map(tag => tag.embedding),
                userMetadataEmbedding.notInterest.tags.map(tag => tag.embedding),
                config.threshold.notInterest
            ),
        };

        // Basic weighted sum score (consider adjusting weights dynamically)
        const contentScore =
            similarity.topic * config.weight.topic +
            similarity.target * config.weight.target +
            similarity.tag * config.weight.tag -
            (similarity.notInterestedTarget + similarity.notInterestedTag) * config.weight.notInterest;

        const scaledRawScore = contentScore * 0.5;

        const finalScore = sigmoid(scaledRawScore, config.sigmoid_k);

        return {
            ...paper,
            scores: {
                topic: similarity.topic,
                target: similarity.target,
                tag: similarity.tag,
                notInterestTarget: similarity.notInterestedTarget,
                notInterestTag: similarity.notInterestedTag,
                final: finalScore,
            }
        };
    });

    const papersMetadataWithScore: PaperMetadataWithScore[] = await Promise.all(scoringPromises);

    return papersMetadataWithScore;
}

const sortPapers = (papers: PaperMetadataWithScore[]) => {
    return papers.sort((a, b) => b.scores.final - a.scores.final);
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

    fs.writeFileSync("result.json", JSON.stringify(sortedPapers.map(paper => {
        const { topic, target, tags, ...rest } = paper;
        return {
            ...rest,
            topic: topic.value,
            target: target.map(target => target.value),
            tags: tags.map(tag => tag.value),
        };
    }), null, 2));

    return sortedPapers;
}

const main = async () => {
    const userMetadata = {
        interest: {
            target: ['web developer', 'data scientist'],
            tags: ["machine learning", "LLM", "healthcare", "social engineering", "IoT"],
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