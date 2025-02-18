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

export const getCosSimilarityMany = (embedding: number[], targets: number[][]) => {
    const similarities = targets.map((target, index) => ({
        similarity: cosineSimilarity(embedding, target),
        index
    }));

    return similarities.filter(({ similarity }) => similarity);
}

export const getCombinedCosSimilarity = (
    embeddings: number[][],
    targets: number[][],
    weights: number[], // 各 target の重み
    threshold = 0,
    alpha = 0.6
  ): number => {
    let weightedMaxSim = -1;       // 重みづけ後の最大値
    let weightedSumSim = 0;        // 重みづけ後の合計値
    let totalWeight = 0;           // 有効なターゲットの重み合計
  
    // ターゲットごとに、embeddings との最大類似度を求める
    for (let i = 0; i < targets.length; i++) {
      const target = targets[i];
      const weight = weights[i] ?? 0; // 安全のため
  
      // このターゲットに対して最も高いコサイン類似度を探す
      let localMax = -1;
      for (const embedding of embeddings) {
        const sim = cosineSimilarity(embedding, target);
        if (sim > localMax) {
          localMax = sim;
        }
      }
  
      // 閾値以上であれば重みを考慮して加算
      if (localMax > threshold && weight > 0) {
        const weightedLocalMax = localMax * weight;
        weightedSumSim += weightedLocalMax;
        totalWeight += weight;
  
        // 最も大きい重み付き類似度を記録
        if (weightedLocalMax > weightedMaxSim) {
          weightedMaxSim = weightedLocalMax;
        }
      }
    }
  
    // 有効なターゲット（thresholdを超えて重みがある）の合計が0なら0を返す
    if (totalWeight === 0) {
      return 0;
    }
  
    // 平均類似度(重み付き)
    const weightedAvgSim = weightedSumSim / totalWeight;
  
    // α 値を使って最大値と平均値をブレンド
    return alpha * weightedMaxSim + (1 - alpha) * weightedAvgSim;
  };