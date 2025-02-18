import { cosineSimilarity } from "ai";

export const getMaxCosSimilarity = (embedding: number[], embeddings: number[][], threshold = 0) => {
    let max = -1;
    for (const target of embeddings) {
        const similarity = cosineSimilarity(embedding, target);
        if (similarity > max && similarity > threshold) {
            max = similarity;
        }
    }

    return max;
}

export const getMaxCosSimilarityMany = (embeddings: number[][], targets: number[][], threshold = 0) => {
    let max = -1;
    for (const target of targets) {
        const similarity = getMaxCosSimilarity(target, embeddings);
        if (similarity > max && similarity > threshold) {
            max = similarity;
        }
    }

    return max;
}

export const getAverageCosSimilarityMany = (embeddings: number[][], targets: number[][], threshold = 0) => {
    let sum = 0;
    for (const target of targets) {
        const similarity = getMaxCosSimilarity(target, embeddings);
        if (similarity > threshold) {
            sum += similarity;
        }
    }

    return sum / targets.length;
}

export const getCombinedCosSimilarity = (
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