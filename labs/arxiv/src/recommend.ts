import { config } from "./config";
import { AITools } from "./libs/ai-tools";
import { scorePapers, sortPapers } from "./score";
import type { UserMetadata } from "./types/user";
import { getArxivPapersWithCache } from "./utils/arxiv";
import fs from 'node:fs';

export const getRecommendedPapers = async (
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

    fs.writeFileSync("result.json", JSON.stringify(papers.map(paper => {
        const { topic, target, tags, ...rest } = paper;
        return {
            ...rest,
            topic: topic.value,
            target: target.map(target => target.value),
            tags: tags.map(tag => tag.value),
        };
    }), null, 2));
}

main();