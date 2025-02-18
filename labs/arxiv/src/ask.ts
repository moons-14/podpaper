import type { UserMetadata, UserMetadataEmbedding } from './types/user';
import fs from 'node:fs';
import "dotenv/config";
import { AITools } from "./libs/ai-tools";
import { getRecommendedPapers } from "./recommend";
import { askConfig } from "./config";
import type { Paper, PaperMetadataEmbedding, TranslatedPaper } from './types/paper';
import { getArxivPapersWithCache } from './libs/arxiv';
import prompts from 'prompts';
import { getPapersMetadata } from './metadata/paper';
import { getPapersMetadataEmbedding } from './embedding/paper';
import { getUserMetadataEmbedding } from './embedding/user';
import { getCosSimilarityMany } from './embedding/similarity';
import { getUserMetadata, saveUserMetadata } from './metadata/user';
import { scorePapers, scorePapersMetadataEmbedding, sortPapers } from './score';

// ユーザーの言語に翻訳する
const translate = async (aiTools: AITools, text: string) => {
    return await aiTools.genText(text, "Please translate the given English into “ja” in plain English. Output should be the translated text only. Do not include anything else.");
}

const translatePapers = async <T extends Paper>(aiTools: AITools, papers: T[], language: string): Promise<TranslatedPaper<T>[]> => {
    const translatedPapers = await Promise.all(papers.map(async paper => {
        const translatedTitle = await translate(aiTools, paper.title);
        const translatedSummary = await translate(aiTools, paper.summary);

        return {
            ...paper,
            translated: {
                title: translatedTitle,
                summary: translatedSummary,
            }
        }
    }));

    return translatedPapers;
}

const askPaperPreference = async <T extends Paper>(paper: TranslatedPaper<T>): Promise<{
    interestPaper?: T,
    notInterestPaper?: T,
}> => {
    console.info("==============================");
    console.info("What is your preference for this paper?");
    console.info(`Title ${paper.translated.title.replace(/\n/g, " ")}`);
    console.info(paper.translated.summary.replace(/\n/g, " "));
    console.info("==============================");

    const { answer } = await prompts({
        type: "select",
        name: "answer",
        message: "Tell us your preferences.",
        choices: [
            { title: "I like this paper.", value: "1" },
            { title: "I don't care.", value: "none" },
            { title: "I don't like this paper.", value: "2" },
        ]
    })

    if (answer === "1") {
        return {
            interestPaper: paper,
        }
    }

    if (answer === "2") {
        return {
            notInterestPaper: paper,
        }
    }

    return {};
}

const getRandomPapers = async (papers: Paper[], number: number) => {
    return papers.sort(() => Math.random() - Math.random()).slice(0, number);
}

const updateEmbeddingItems = (
    itemsToAdd: { value: string; weight: number; embedding: number[] }[],
    existingItems: { value: string; weight: number; embedding: number[] }[],
    thresholdMatch: number,
    thresholdRelated: number,
    weightMatch: number,
    weightRelated: number
) => {
    for (const item of itemsToAdd) {
        const similarityList = getCosSimilarityMany(
            item.embedding,
            existingItems.map(i => i.embedding)
        );

        let shouldAddItem = true;
        let weightRelated = 1;

        for (const [index, similarity] of similarityList.entries()) {
            if (similarity.similarity > thresholdMatch) {
                // 既存アイテムの重みをさらに強める
                existingItems[index].weight *= weightMatch;
                shouldAddItem = false;
                break;
            }
            if (similarity.similarity > thresholdRelated) {
                // 関連度がそこそこあるならちょっとだけ重みを強める
                existingItems[index].weight *= weightRelated;
                weightRelated *= weightRelated;
                break;
            }
        }

        if (shouldAddItem) {
            existingItems.push({
                ...item,
                weight: weightRelated
            });
        }
    }
}

const normalizeWeights = (
    items: { value: string; weight: number; embedding: number[] }[]
) => {
    if (items.length === 0) return;
    const maxWeight = Math.max(...items.map(i => i.weight));
    if (maxWeight > 0) {
        for (const i of items) {
            i.weight = i.weight / maxWeight;
        }
    }
}

const updateUserMetadataEmbedding = (
    currentMetadata: UserMetadataEmbedding,
    selectedPaper: PaperMetadataEmbedding,
    isInterest: boolean
) => {
    const { tags: paperTags, target: paperTargets } = selectedPaper;

    if (!paperTags || !paperTargets) {
        return currentMetadata;
    }

    const {
        threshold: { match: thresholdMatch, related: thresholdRelated },
        weight: { match: weightMatch, related: weightRelated }
    } = askConfig;

    // 更新先となる配列を取得（興味あり or 興味なし）
    const targetTags = isInterest
        ? currentMetadata.interest.tags
        : currentMetadata.notInterest.tags;
    const targetTargets = isInterest
        ? currentMetadata.interest.target
        : currentMetadata.notInterest.target;

    // 追加するアイテムを作る
    const newTags = paperTags.map(tag => ({
        value: tag.value,
        weight: 1,
        embedding: tag.embedding,
    }));

    const newTargets = paperTargets.map(t => ({
        value: t.value,
        weight: 1,
        embedding: t.embedding,
    }));

    // タグ更新
    updateEmbeddingItems(newTags, targetTags, thresholdMatch, thresholdRelated, weightMatch, weightRelated);
    // ターゲット更新
    updateEmbeddingItems(newTargets, targetTargets, thresholdMatch, thresholdRelated, weightMatch, weightRelated);

    // 重み正規化
    normalizeWeights(targetTags);
    normalizeWeights(targetTargets);

    return currentMetadata;
}

const askRandomQuestions = async (aiTools: AITools, questionCount: number, language: string, queryCategory: string, timeFilterMS: number, userMetadata?: UserMetadata) => {
    const papers = await getArxivPapersWithCache(queryCategory, timeFilterMS);

    const randomPapers = await getRandomPapers(papers, questionCount);

    const papersMetadata = await getPapersMetadata(aiTools, randomPapers, userMetadata);
    const papersMetadataEmbedding = await getPapersMetadataEmbedding(aiTools, papersMetadata);

    const translatedPapers = await translatePapers(aiTools, papersMetadataEmbedding, language);

    const userMetadataEmbedding = userMetadata && await getUserMetadataEmbedding(aiTools, userMetadata);

    let currentUserMetadataEmbedding = userMetadataEmbedding
        ? userMetadataEmbedding
        : {
            interest: {
                tags: [],
                target: []
            },
            notInterest: {
                tags: [],
                target: []
            }
        };

    for (let i = 0; i < questionCount; i++) {
        const paper = translatedPapers[i];

        // ユーザーに質問
        const { interestPaper, notInterestPaper } = await askPaperPreference(paper);

        // 好みのアップデート処理
        if (interestPaper) {
            currentUserMetadataEmbedding = updateUserMetadataEmbedding(
                currentUserMetadataEmbedding,
                interestPaper,
                true
            );
        }
        if (notInterestPaper) {
            currentUserMetadataEmbedding = updateUserMetadataEmbedding(
                currentUserMetadataEmbedding,
                notInterestPaper,
                false
            );
        }
    }

    return currentUserMetadataEmbedding;
}

// 既存のユーザーメタデータを更新する

const askSortedQuestions = async (aiTools: AITools, questionCount: number, language: string, queryCategory: string, timeFilterMS: number, userMetadata: UserMetadata) => {
    const papers = await getArxivPapersWithCache(queryCategory, timeFilterMS);
    const papersMetadata = await getPapersMetadata(aiTools, papers, userMetadata);
    const papersMetadataEmbedding = await getPapersMetadataEmbedding(aiTools, papersMetadata);

    let currentMetadataEmbedding = await getUserMetadataEmbedding(aiTools, userMetadata);

    for (let i = 0; i < questionCount; i++) {
        const scoredPapers = await scorePapersMetadataEmbedding(currentMetadataEmbedding, papersMetadataEmbedding);

        const sortedPapers = sortPapers(scoredPapers);

        const questionPapers = sortedPapers.slice(0, Math.ceil(sortedPapers.length / 4));

        const randomPaper = questionPapers[Math.floor(Math.random() * questionPapers.length)];

        const translatedPaper = await translatePapers(aiTools, [randomPaper], language);

        const { interestPaper, notInterestPaper } = await askPaperPreference(translatedPaper[0]);

        if (interestPaper) {
            currentMetadataEmbedding = updateUserMetadataEmbedding(
                currentMetadataEmbedding,
                interestPaper,
                true
            );
        }

        if (notInterestPaper) {
            currentMetadataEmbedding = updateUserMetadataEmbedding(
                currentMetadataEmbedding,
                notInterestPaper,
                false
            );
        }
    }

    return currentMetadataEmbedding;
}

const main = async () => {

    const aiTools = new AITools(process.env.AI_STUDIO_API_KEY || "", process.env.AI_STUDIO_BASE_URL);

    const userMetadata = await askRandomQuestions(aiTools, askConfig.questionCount.random, askConfig.language, askConfig.queryCategory, askConfig.timeFilterMS);

    // const userMetadata = await getUserMetadata();

    if (!userMetadata) return;

    await askSortedQuestions(aiTools, askConfig.questionCount.sorted, askConfig.language, askConfig.queryCategory, askConfig.timeFilterMS, userMetadata);

    await saveUserMetadata(userMetadata);

};

main();