import type { AITools } from "../libs/ai-tool";
import type { UserMetadata, UserMetadataEmbedding } from "../types/user";

export const getUserMetadataEmbedding = async (
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