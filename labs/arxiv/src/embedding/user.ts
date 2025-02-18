import type { AITools } from "../libs/ai-tools";
import type { UserMetadata, UserMetadataEmbedding } from "../types/user";

export const getUserMetadataEmbedding = async (
    aiTools: AITools,
    metadata: UserMetadata
): Promise<UserMetadataEmbedding> => {
    console.debug("getting user metadata embedding...");

    const allEmbeddingText = [...new Set([...metadata.interest.target.map(v => v.value), ...metadata.notInterest.target.map(v => v.value), ...metadata.interest.tags.map(v => v.value), ...metadata.notInterest.tags.map(v => v.value)])];
    const embeddings = await aiTools.getMultiEmbedding(allEmbeddingText);

    const userMetadataEmbedding: UserMetadataEmbedding = {
        interest: {
            target: metadata.interest.target.map((value) => {
                const embedding = embeddings.find(e => e.value === value.value);
                return embedding ? { embedding: embedding.embedding, value: embedding.value, weight: value.weight } : null;
            }).filter((v) => v !== null),
            tags: metadata.interest.tags.map((value) => {
                const embedding = embeddings.find(e => e.value === value.value);
                return embedding ? { embedding: embedding.embedding, value: embedding.value, weight: value.weight } : null;
            }).filter((v) => v !== null),
        },
        notInterest: {
            target: metadata.notInterest.target.map((value) => {
                const embedding = embeddings.find(e => e.value === value.value);
                return embedding ? { embedding: embedding.embedding, value: embedding.value, weight: value.weight } : null;
            }).filter((v) => v !== null),
            tags: metadata.notInterest.tags.map((value) => {
                const embedding = embeddings.find(e => e.value === value.value);
                return embedding ? { embedding: embedding.embedding, value: embedding.value, weight: value.weight } : null;
            }).filter((v) => v !== null),
        }
    }

    console.debug("got user metadata embedding");
    return userMetadataEmbedding;
}