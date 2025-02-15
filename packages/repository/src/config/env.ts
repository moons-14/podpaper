if (!process.env.DB_FILE_NAME) {
    throw new Error('DB_FILE_NAME is not defined');
}

export const ENV = {
    DB_FILE_NAME: process.env.DB_FILE_NAME,
}