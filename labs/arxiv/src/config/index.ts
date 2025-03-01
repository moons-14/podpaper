export const recommendConfig: {
    threshold: {
        interest: number,
        notInterest: number,
    },
    weight: {
        topic: number,
        target: number,
        tag: number,
        notInterest: number
    },
    language: string,
    timeFilterMS: number,
    sigmoid_k: number,
    queryCategory: string
} = {
    threshold: {
        interest: 0.35,
        notInterest: 0.6,
    },
    weight: {
        topic: 4,
        target: 2,
        tag: 3,
        notInterest: 2.0
    },
    language: "ja",
    timeFilterMS: 1000 * 60 * 60 * 24 * 4, // 4 day
    sigmoid_k: 0.5,
    queryCategory: "cat:cs.* OR cat:econ.* OR cat:eess.* OR cat:math.* OR cat:astro-ph.* OR cat:cond-mat.* OR cat:gr-qc OR cat:hep-ex OR cat:hep-lat OR cat:hep-ph OR cat:hep-th OR cat:math-ph OR cat:nlin.* OR cat:nucl-ex OR cat:nucl-th OR cat:physics.* OR cat:quant-ph OR cat:q-bio.* OR cat:q-fin.* OR cat:stat.*"
}

export const askConfig: {
    questionCount: {
        random: number,
        sorted: number
    }
    language: string
    timeFilterMS: number
    queryCategory: string
    threshold: {
        match: number
        related: number
    },
    weight: {
        match: number
        related: number
    }
} = {
    questionCount:{
        random: 20,
        sorted: 5
    },
    language: "ja",
    timeFilterMS: 1000 * 60 * 60 * 24 * 4, // 4 day
    queryCategory: "cat:cs.* OR cat:econ.* OR cat:eess.* OR cat:math.* OR cat:astro-ph.* OR cat:cond-mat.* OR cat:gr-qc OR cat:hep-ex OR cat:hep-lat OR cat:hep-ph OR cat:hep-th OR cat:math-ph OR cat:nlin.* OR cat:nucl-ex OR cat:nucl-th OR cat:physics.* OR cat:quant-ph OR cat:q-bio.* OR cat:q-fin.* OR cat:stat.*",
    threshold: {
        match: 0.8,
        related: 0.4
    },
    weight: {
        match: 1.2,
        related: 1.05
    }
}