import type { AITools } from "../libs/ai-tools";
import type { PaperMetadata, PaperMetadataEmbedding } from "../types/paper";

export const getPapersMetadataEmbedding = async (
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