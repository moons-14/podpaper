import fs from "node:fs";
import Parser from "rss-parser";
import type { Paper } from "../types/paper";

export const getArxivPapers = async (
    query: string,
    timeFilterMS: number,
) => {
    const papers: Paper[] = [];

    const parser = new Parser<{
        feedUrl: string,
        title: string,
        lastBuildDate: string,
        link: string,
    }, {
        title: string,
        link: string,
        pubDate: string,
        author: string,
        summary: string,
        id: string,
        isoDate: string,
    }>();

    const now = new Date();
    const from = new Date(now.getTime() - timeFilterMS);
    from.setHours(from.getHours() - 5);
    now.setHours(now.getHours() - 5);
    const fromStr = `${from.getFullYear()}${String(from.getMonth() + 1).padStart(2, "0")}${String(from.getDate()).padStart(2, "0")}0000`;
    const nowStr = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}0000`;
    const submittedDate = `[${fromStr}+TO+${nowStr}]`;

    const search_query = `${encodeURIComponent(`(${query})`)}+AND+submittedDate:${submittedDate}`;

    while (true) {

        const url = `https://export.arxiv.org/api/query?search_query=${search_query}&max_results=500&start=${papers.length}`;

        const paper = await parser.parseURL(url);

        if (paper.items.length === 0) {
            break;
        }

        for (const item of paper.items) {
            papers.push({
                author: item.author,
                title: item.title,
                link: item.link,
                summary: item.summary,
                id: item.id,
                isoDate: item.isoDate,
                pubDate: new Date(item.isoDate),
            });
        }

    }

    return papers;
}

export const getArxivPapersWithCache = async (
    query: string,
    timeFilterMS: number
): Promise<Paper[]> => {
    if (!fs.existsSync("./cache")) {
        fs.mkdirSync("./cache");
    }

    const now = new Date();
    const nowStr = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}`;

    const cachePath = `./cache/${nowStr}-${timeFilterMS}.json`;

    if (fs.existsSync(cachePath)) {
        const data = fs.readFileSync(cachePath, "utf-8");
        const papers = JSON.parse(data);

        console.debug("cache hit");
        console.debug("cache path", cachePath);
        console.debug("papers length", papers.length);

        return papers;
    }

    console.debug("cache miss");
    console.debug("getting papers....");

    const papers = await getArxivPapers(query, timeFilterMS);

    console.debug("got papers");
    console.debug("papers length", papers.length);

    fs.writeFileSync(cachePath, JSON.stringify(papers));

    return papers;
}