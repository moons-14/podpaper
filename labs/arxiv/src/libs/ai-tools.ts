import { createGoogleGenerativeAI, type GoogleGenerativeAIProvider } from "@ai-sdk/google";
import { embed, embedMany, generateObject, type Schema } from "ai";
import type { z } from "zod";

export class AITools {
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