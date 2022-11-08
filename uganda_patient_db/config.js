const env = process.env;

const config = {
  db: { /* do not put password or any sensitive info here, done only for demo */
    host: env.DB_HOST || 'kashin.db.elephantsql.com',
    port: env.DB_PORT || '5432',
    user: env.DB_USER || 'hrkteijk',
    password: env.DB_PASSWORD || '0QTXjY5sVQevRazatLdYIwNJ3sYo5N8f',
   database: env.DB_NAME || 'hrkteijk',
  },
  listPerPage: env.LIST_PER_PAGE || 10,
};

module.exports = config;
