export type UserMetadata = {
    interest: {
        target: {
            value: string,
            weight: number,
        }[],
        tags: {
            value: string,
            weight: number,
        }[],
    },
    notInterest: {
        target: {
            value: string,
            weight: number,
        }[],
        tags: {
            value: string,
            weight: number,
        }[],
    }
}

export type UserMetadataEmbedding = {
    interest: {
        target: {
            embedding: number[],
            value: string,
            weight: number,
        }[],
        tags: {
            embedding: number[],
            value: string,
            weight: number,
        }[],
    },
    notInterest: {
        target: {
            embedding: number[],
            value: string,
            weight: number,
        }[],
        tags: {
            embedding: number[],
            value: string,
            weight: number,
        }[],
    }
}