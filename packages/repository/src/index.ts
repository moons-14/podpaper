import 'dotenv/config';
import { drizzle } from 'drizzle-orm/libsql';
import { ENV } from './config/env';
import { arxivPapersTable } from './db/schema';


const db = drizzle(ENV.DB_FILE_NAME);

// arxivの論文を追加する
export const insertArxivPaper = async ({
    creator,
    rights,
    title,
    link,
    pubDate,
    content,
    contentSnippet,
    guid,
    categories,
}: {
    creator: string[],
    rights: string,
    title: string,
    link: string,
    pubDate: Date,
    content: string,
    contentSnippet: string,
    guid: string,
    categories: string[],
}) => {

    const paper: typeof arxivPapersTable.$inferInsert = {
        id: guid,
        rights,
        title,
        link,
        pubDate: pubDate.toISOString(),
        content,
        contentSnippet,
        creator: creator.join(','),
        categories: categories.join(','),
    }

    await db.insert(arxivPapersTable).values(paper).onConflictDoUpdate({ target: arxivPapersTable.id, set: paper });

}