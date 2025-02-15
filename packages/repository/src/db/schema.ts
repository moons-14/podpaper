import { int, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const arxivPapersTable = sqliteTable("arxiv_papers", {
    id: text().notNull().primaryKey(),
    rights: text().notNull(),
    title: text().notNull(),
    link: text().notNull(),
    pubDate: text().notNull(),
    content: text().notNull(),
    contentSnippet: text().notNull(),
    creator: text().notNull(),
    categories: text().notNull(),
});
