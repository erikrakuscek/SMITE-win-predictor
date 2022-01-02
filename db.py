import psycopg2


class DataBase:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="",
            port="",
            database="",
            user="",
            password=""
        )
        self.conn.autocommit = True

    def insert(self, query, params):
        cur = self.conn.cursor()
        cur.execute(query, params)
        cur.close()
        return True

    def empty_database(self):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM match")
        cur.close()
        return True

    def create_tables(self):
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS Match (" +
                    "id SERIAL PRIMARY KEY," +
                    "t1_god1 INT," +
                    "t1_god2 INT," +
                    "t1_god3 INT," +
                    "t1_god4 INT," +
                    "t1_god5 INT," +
                    "t1_player1 NUMERIC," +
                    "t1_player2 NUMERIC," +
                    "t1_player3 NUMERIC," +
                    "t1_player4 NUMERIC," +
                    "t1_player5 NUMERIC," +
                    "t1_player1_mmr NUMERIC," +
                    "t1_player2_mmr NUMERIC," +
                    "t1_player3_mmr NUMERIC," +
                    "t1_player4_mmr NUMERIC," +
                    "t1_player5_mmr NUMERIC," +
                    "t1_ban1 INT," +
                    "t1_ban2 INT," +
                    "t1_ban3 INT," +
                    "t2_god1 INT," +
                    "t2_god2 INT," +
                    "t2_god3 INT," +
                    "t2_god4 INT," +
                    "t2_god5 INT," +
                    "t2_player1 NUMERIC," +
                    "t2_player2 NUMERIC," +
                    "t2_player3 NUMERIC," +
                    "t2_player4 NUMERIC," +
                    "t2_player5 NUMERIC," +
                    "t2_player1_mmr NUMERIC," +
                    "t2_player2_mmr NUMERIC," +
                    "t2_player3_mmr NUMERIC," +
                    "t2_player4_mmr NUMERIC," +
                    "t2_player5_mmr NUMERIC," +
                    "t2_ban1 INT," +
                    "t2_ban2 INT," +
                    "t2_ban3 INT," +
                    "t2_ban4 INT," +
                    "t2_ban5 INT," +
                    "win INT"
                    ");")
        cur.execute("CREATE TABLE IF NOT EXISTS God (" +
                    "id INT PRIMARY KEY," +
                    "name TEXT" +
                    ");")
        cur.close()
        return True

    def select(self, query, params):
        cur = self.conn.cursor()
        cur.execute(query, params)
        value = cur.fetchall()
        cur.close()
        return value
