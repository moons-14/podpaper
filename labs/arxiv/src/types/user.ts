export type UserMetadata = {
    interest: {
        target: string[],
        tags: string[],
    },
    notInterest: {
        target: string[],
        tags: string[],
    }
}

export type UserMetadataEmbedding = {
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