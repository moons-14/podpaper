export type Paper = {
    author: string,
    title: string,
    link: string,
    summary: string,
    id: string,
    isoDate: string,
    pubDate: Date,
}

export type PaperMetadata = {
    topic: string,
    target: string[],
    tags: string[],
    type: string,
} & Paper

export type PaperMetadataEmbedding = (Omit<PaperMetadata, "topic" | "target" | "tags"> & {
    topic: {
        embedding: number[],
        value: string,
    },
    target: {
        embedding: number[],
        value: string,
    }[],
    tags: {
        embedding: number[],
        value: string,
    }[],
})

export type PaperMetadataScore = {
    topic: number,
    target: number,
    tag: number,
    notInterestTarget: number,
    notInterestTag: number,
    final: number,
}

export type PaperMetadataWithScore = PaperMetadataEmbedding & {
    scores: PaperMetadataScore
}

export type TranslatedPaper<T> ={
    translated: {
        title: string,
        summary: string,
    }
} & T