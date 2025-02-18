import fs from "node:fs";
import type { UserMetadata } from "../types/user";

export const getUserMetadata = async (): Promise<UserMetadata | undefined> => {
    if (!fs.existsSync("./cache")) {
        fs.mkdirSync("./cache");
    }

    if (!fs.existsSync("./cache/user-metadata.json")) {
        return undefined;
    }

    const userMetadata = JSON.parse(fs.readFileSync("./cache/user-metadata.json", "utf-8")) as UserMetadata;

    return userMetadata;
}

export const saveUserMetadata = async (userMetadata: UserMetadata) => {
    if (!fs.existsSync("./cache")) {
        fs.mkdirSync("./cache");
    }

    if (fs.existsSync("./cache/user-metadata.json")) {
        fs.unlinkSync("./cache/user-metadata.json");
    }

    // embeddingが含まれているなら除外する
    const userMetadataWithoutEmbedding: UserMetadata = {
        ...userMetadata,
        interest: {
            target: userMetadata.interest.target.map(v => ({ value: v.value, weight: v.weight })),
            tags: userMetadata.interest.tags.map(v => ({ value: v.value, weight: v.weight })),
        },
        notInterest: {
            target: userMetadata.notInterest.target.map(v => ({ value: v.value, weight: v.weight })),
            tags: userMetadata.notInterest.tags.map(v => ({ value: v.value, weight: v.weight })),
        }
    }
    fs.writeFileSync("./cache/user-metadata.json", JSON.stringify(userMetadataWithoutEmbedding, null, 4));
}