import type { AITools } from "../libs/ai-tools";
import type { UserMetadata } from "../types/user";
import type { Paper, PaperMetadata } from "../types/paper";
import { z } from "zod";

export const getPaperMetadata = async (
    aiTools: AITools,
    paper: Paper,
    userMetadata?: UserMetadata
): Promise<PaperMetadata | null> => {

    const pickupUserMetadata = userMetadata && {
        interest: {
            tags: userMetadata.interest.tags.sort(() => Math.random() - 0.5).slice(0, 10).map(tag => tag.value.toLowerCase().trim()),
            target: userMetadata.interest.target.sort(() => Math.random() - 0.5).slice(0, 10).map(target => target.value.toLowerCase().trim())
        },
        notInterest: {
            tags: userMetadata.notInterest.tags.sort(() => Math.random() - 0.5).slice(0, 10).map(tag => tag.value.toLowerCase().trim()),
            target: userMetadata.notInterest.target.sort(() => Math.random() - 0.5).slice(0, 10).map(target => target.value.toLowerCase().trim())
        }
    }

    const title = paper.title;
    const summary = paper.summary;

    const metadata = await aiTools.genObject(
        `Please analyze the content based on the names and summaries of the following papers and generate JSON with meta-information.

# Instructions
Based on the information in the following papers, please summarize the tags, targets, keywords, topics, and types of papers. Please output the results in JSON format.
please do not use words that are too common or words that can be abbreviated to obscure the context. Please tag them reliably and clearly.
${pickupUserMetadata && `However, please see the following example of a user's tag and be aware of the relative nature of the output to that tag.

## User's tag
- InterestedTarget: ${pickupUserMetadata.interest.target.join(", ")}
- InterestedTags: ${pickupUserMetadata.interest.tags.join(", ")}
- notInterestedTarget: ${pickupUserMetadata.notInterest.target.join(", ")}
- notInterestedTags: ${pickupUserMetadata.notInterest.tags.join(", ")}`}

## Article Title.
${title}

## Abstract of the paper
${summary.replace(/\n/g, " ")}`,
        z.object({
            tags: z.array(z.string()).describe("Please tag words that are important in the paper and that categorize the paper."),
            target: z.array(z.string()).describe("Please output the professions that will be most affected by this paper. E.g.) web engineer, infrastructure engineer, janitor"),
            topic: z.string().describe("Please list the single most important topic in the paper."),
            type: z.enum(["empirical", "theoretical", "literature", "experimental", "simulation"]).describe("Please indicate whether the paper is an empirical study/ theoretical study/ literature review/ experimental study/simulation."),
        })
    );

    if (metadata === null) {
        return null;
    }

    return {
        ...paper,
        ...metadata,
    }
}

export const getPapersMetadata = async (
    aiTools: AITools,
    papers: Paper[],
    userMetadata?: UserMetadata
): Promise<PaperMetadata[]> => {
    console.debug("getting papers metadata...");

    const metadataPromises = papers.map(paper => getPaperMetadata(aiTools, paper, userMetadata));
    const results = await Promise.all(metadataPromises);

    const paperMetadata = results.filter(metadata => metadata !== null) as PaperMetadata[];

    console.debug("got papers metadata");
    console.debug("papers metadata length", paperMetadata.length);

    return paperMetadata;
};