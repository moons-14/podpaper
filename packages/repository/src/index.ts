import 'dotenv/config';
import { drizzle } from 'drizzle-orm/libsql';
import { ENV } from './config/env';


const db = drizzle(ENV.DB_FILE_NAME);
