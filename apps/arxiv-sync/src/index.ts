import Parser from 'rss-parser';
import { insertArxivPaper } from "@podpaper/repository";

const parser = new Parser<{
    feedUrl: string,
    paginationLinks: {
        self: string,
    },
    title: string,
    description: string,
    pubDate: string,
    managingEditor: string,
    link: string,
    language: string,
    lastBuildDate: string,
    docs: string,
    skipDays: {
        day: string[],
    }
}, {
    creator: string,
    rights: string,
    title: string,
    link: string,
    pubDate: string,
    "dc:creator": string,
    content: string,
    contentSnippet: string,
    guid: string,
    categories: string[],
    isoDate: string,
}>();

const allCategories = [
    'astro-ph',
    'cond-mat',
    'cs',
    'econ',
    'eess',
    'gr-qc',
    'hep-ex',
    'hep-lat',
    'hep-ph',
    'hep-th',
    'math',
    'math-ph',
    'nlin',
    'nucl-ex',
    'nucl-th',
    'physics',
    'q-bio',
    'q-fin',
    'quant-ph',
    'stat'
]

const main = async () => {

    let progress = 0;
    for (const category of allCategories) {
        const feed = await parser.parseURL(`https://rss.arxiv.org/rss/${category}`);

        for (const item of feed.items) {
            const creator = item.creator.split(', ');
            const rights = item.rights;
            const title = item.title;
            const link = item.link;
            const pubDate = new Date(item.pubDate);
            const content = item.content;
            const contentSnippet = item.contentSnippet;
            const guid = item.guid;
            const categories = item.categories;

            await insertArxivPaper({
                creator,
                rights,
                title,
                link,
                pubDate,
                content,
                contentSnippet,
                guid,
                categories,
            });

            progress++;
            if (progress % 100 === 0) {
                console.log("progress", progress);
            }
        }
    }
}

main()