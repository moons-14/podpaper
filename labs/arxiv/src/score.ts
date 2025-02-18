import { recommendConfig } from "./config";
import { getPapersMetadataEmbedding } from "./embedding/paper";
import { getCombinedCosSimilarity } from "./embedding/similarity";
import { getUserMetadataEmbedding } from "./embedding/user";
import type { AITools } from "./libs/ai-tools";
import { getPapersMetadata } from "./metadata/paper";
import type { Paper, PaperMetadataEmbedding, PaperMetadataWithScore } from "./types/paper";
import type { UserMetadata, UserMetadataEmbedding } from "./types/user";
import { sigmoid } from "./utils/sigmoid";

export const scorePapers = async (
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
                userMetadataEmbedding.interest.tags.map(tag => tag.weight),
                recommendConfig.threshold.interest
            ),
            target: getCombinedCosSimilarity(
                paper.target.map(target => target.embedding),
                userMetadataEmbedding.interest.target.map(target => target.embedding),
                userMetadataEmbedding.interest.target.map(target => target.weight),
                recommendConfig.threshold.interest
            ),
            tag: getCombinedCosSimilarity(
                paper.tags.map(tag => tag.embedding),
                userMetadataEmbedding.interest.tags.map(tag => tag.embedding),
                userMetadataEmbedding.interest.tags.map(tag => tag.weight),
                recommendConfig.threshold.interest
            ),
            notInterestedTarget: getCombinedCosSimilarity(
                paper.target.map(target => target.embedding),
                userMetadataEmbedding.notInterest.target.map(target => target.embedding),
                userMetadataEmbedding.notInterest.target.map(target => target.weight),
                recommendConfig.threshold.notInterest
            ),
            notInterestedTag: getCombinedCosSimilarity(
                paper.tags.map(tag => tag.embedding),
                userMetadataEmbedding.notInterest.tags.map(tag => tag.embedding),
                userMetadataEmbedding.notInterest.tags.map(tag => tag.weight),
                recommendConfig.threshold.notInterest
            ),
        };

        // Basic weighted sum score (consider adjusting weights dynamically)
        const contentScore =
            similarity.topic * recommendConfig.weight.topic +
            similarity.target * recommendConfig.weight.target +
            similarity.tag * recommendConfig.weight.tag -
            (similarity.notInterestedTarget + similarity.notInterestedTag) * recommendConfig.weight.notInterest;

        const scaledRawScore = contentScore * 0.5;

        const finalScore = sigmoid(scaledRawScore, recommendConfig.sigmoid_k);

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

export const scorePapersMetadataEmbedding = async (
    userMetadataEmbedding: UserMetadataEmbedding,
    papersMetadataEmbedding: PaperMetadataEmbedding[]
) => {
    const scoringPromises = papersMetadataEmbedding.map(async (paper) => {
        const similarity = {
            topic: getCombinedCosSimilarity(
                [paper.topic.embedding],
                userMetadataEmbedding.interest.tags.map(tag => tag.embedding),
                userMetadataEmbedding.interest.tags.map(tag => tag.weight),
                recommendConfig.threshold.interest
            ),
            target: getCombinedCosSimilarity(
                paper.target.map(target => target.embedding),
                userMetadataEmbedding.interest.target.map(target => target.embedding),
                userMetadataEmbedding.interest.target.map(target => target.weight),
                recommendConfig.threshold.interest
            ),
            tag: getCombinedCosSimilarity(
                paper.tags.map(tag => tag.embedding),
                userMetadataEmbedding.interest.tags.map(tag => tag.embedding),
                userMetadataEmbedding.interest.tags.map(tag => tag.weight),
                recommendConfig.threshold.interest
            ),
            notInterestedTarget: getCombinedCosSimilarity(
                paper.target.map(target => target.embedding),
                userMetadataEmbedding.notInterest.target.map(target => target.embedding),
                userMetadataEmbedding.notInterest.target.map(target => target.weight),
                recommendConfig.threshold.notInterest
            ),
            notInterestedTag: getCombinedCosSimilarity(
                paper.tags.map(tag => tag.embedding),
                userMetadataEmbedding.notInterest.tags.map(tag => tag.embedding),
                userMetadataEmbedding.notInterest.tags.map(tag => tag.weight),
                recommendConfig.threshold.notInterest
            ),
        };

        // Basic weighted sum score (consider adjusting weights dynamically)
        const contentScore =
            similarity.topic * recommendConfig.weight.topic +
            similarity.target * recommendConfig.weight.target +
            similarity.tag * recommendConfig.weight.tag -
            (similarity.notInterestedTarget + similarity.notInterestedTag) * recommendConfig.weight.notInterest;

        const scaledRawScore = contentScore * 0.5;

        const finalScore = sigmoid(scaledRawScore, recommendConfig.sigmoid_k);

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

export const sortPapers = (papers: PaperMetadataWithScore[]) => {
    return papers.sort((a, b) => b.scores.final - a.scores.final);
}