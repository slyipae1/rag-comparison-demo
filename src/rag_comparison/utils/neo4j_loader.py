import os
import json
from neo4j import GraphDatabase, exceptions
import re
import ast

NEO4J_SERVER_URI = os.getenv("NEO4J_SERVER_URI", "http://localhost:7474")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
target_db_name = os.getenv("NEO4J_DATABASE", "neo4j")

class Neo4jLoader:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        print(f"准备连接到 Neo4j 数据库 '{self.database}'...")

    def close(self):
        if self.driver:
            self.driver.close()
            print("已断开与 Neo4j 的连接。")

    def run_query(self, query, parameters=None):
        try:
            records, _, _ = self.driver.execute_query(query, parameters, database=self.database)
            return records
        except Exception as e:
            print(f"执行查询时出错: {query}\n错误详情: {e}")
            return None

    @staticmethod
    def _sanitize_entity_type(entity_type):
        # 允许汉字、字母、数字和下划线，移除其他特殊字符
        return re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fa5]', '_', entity_type.strip())

    # @staticmethod
    # def _create_entities_and_relations(tx, entities, relations, import_id):
    #     # 创建实体（支持list或dict格式）
    #     for entity in entities:
    #         if isinstance(entity, dict):
    #             entity_id = entity.get("id") or entity.get("name")
    #             entity_type = entity.get("type") or entity.get("entity_type", "Entity")
    #             props = {k: v for k, v in entity.items() if k not in ("id", "type", "entity_type")}
    #         elif isinstance(entity, (list, tuple)) and len(entity) >= 2:
    #             entity_id, entity_type = entity[0], entity[1]
    #             props = {}
    #         else:
    #             continue
    #         sanitized_type = Neo4jLoader._sanitize_entity_type(entity_type)
    #         # 动态构建属性SET语句
    #         set_props = ", ".join([f"n.{k} = ${k}" for k in props])
    #         query = (
    #             f"MERGE (n:{sanitized_type} {{name: $name}}) "
    #             "SET n.import_ids = CASE "
    #             "WHEN n.import_ids IS NULL THEN [$import_id] "
    #             "WHEN NOT $import_id IN n.import_ids THEN n.import_ids + $import_id "
    #             "ELSE n.import_ids END"
    #         )
    #         if set_props:
    #             query += f", {set_props}"
    #         params = {"name": entity_id, "import_id": import_id, **props}
    #         tx.run(query, **params)

    #     # 创建关系（支持list或dict格式）
    #     for rel in relations:
    #         if isinstance(rel, dict):
    #             head = rel.get("head") or rel.get("source")
    #             tail = rel.get("tail") or rel.get("target")
    #             rel_type = rel.get("type") or rel.get("rel_type") or rel.get("keywords", "related_to")
    #             props = {k: v for k, v in rel.items() if k not in ("head", "source", "tail", "target", "type", "rel_type", "keywords")}
    #         elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
    #             head, rel_type, tail = rel[0], rel[1], rel[2]
    #             props = {}
    #         else:
    #             continue
    #         set_props = ", ".join([f"r.{k} = ${k}" for k in props])
    #         query = (
    #             "MATCH (h {name: $head_name}), (t {name: $tail_name}) "
    #             f"MERGE (h)-[r:`{rel_type}`]->(t) "
    #             "SET r.import_ids = CASE "
    #             "WHEN r.import_ids IS NULL THEN [$import_id] "
    #             "WHEN NOT $import_id IN r.import_ids THEN r.import_ids + $import_id "
    #             "ELSE r.import_ids END"
    #         )
    #         if set_props:
    #             query += f", {set_props}"
    #         params = {"head_name": head, "tail_name": tail, "import_id": import_id, **props}
    #         try:
    #             tx.run(query, **params)
    #         except exceptions.ClientError as e:
    #             print(f"创建关系失败: {head} -[{rel_type}]-> {tail} - {e}")
    
    @staticmethod
    def _create_entities_and_relations(tx, entities, relations, import_id):
        # 步骤 1: 创建一个实体名称到实体类型的映射字典，用于后续关系匹配
        entity_type_map = {}
        for entity in entities:
            if isinstance(entity, dict):
                entity_id = entity.get("id") or entity.get("name")
                entity_type = entity.get("type") or entity.get("entity_type", "Entity")
            elif isinstance(entity, (list, tuple)) and len(entity) >= 2:
                entity_id, entity_type = entity[0], entity[1]
            else:
                continue
            if entity_id and entity_type:
                entity_type_map[entity_id] = entity_type

        # 创建实体（与原逻辑保持不变）
        for entity in entities:
            if isinstance(entity, dict):
                entity_id = entity.get("id") or entity.get("name")
                entity_type = entity.get("type") or entity.get("entity_type", "Entity")
                props = {k: v for k, v in entity.items() if k not in ("id", "type", "entity_type")}
            elif isinstance(entity, (list, tuple)) and len(entity) >= 2:
                entity_id, entity_type = entity[0], entity[1]
                props = {}
            else:
                continue
            
            sanitized_type = Neo4jLoader._sanitize_entity_type(entity_type)
            set_props = ", ".join([f"n.{k} = ${k}" for k in props])
            query = (
                f"MERGE (n:{sanitized_type} {{name: $name}}) "
                "SET n.import_ids = CASE "
                "WHEN n.import_ids IS NULL THEN [$import_id] "
                "WHEN NOT $import_id IN n.import_ids THEN n.import_ids + $import_id "
                "ELSE n.import_ids END"
            )
            if set_props:
                query += f", {set_props}"
            params = {"name": entity_id, "import_id": import_id, **props}
            tx.run(query, **params)

        # 步骤 2: 创建关系，使用更精确的匹配逻辑
        for rel in relations:
            if isinstance(rel, dict):
                head = rel.get("head") or rel.get("source")
                tail = rel.get("tail") or rel.get("target")
                rel_type = rel.get("type") or rel.get("rel_type") or rel.get("keywords", "related_to")
                props = {k: v for k, v in rel.items() if k not in ("head", "source", "tail", "target", "type", "rel_type", "keywords")}
            elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
                head, rel_type, tail = rel[0], rel[1], rel[2]
                props = {}
            else:
                continue
            
            # 从映射中查找头尾实体的类型
            head_type = entity_type_map.get(head)
            tail_type = entity_type_map.get(tail)

            # 如果在当前批次的实体中找不到某个实体类型，打印警告并跳过，避免错误
            if not head_type or not tail_type:
                print(f"警告: 无法为关系 '{head} -[{rel_type}]-> {tail}' 找到明确的实体类型，已跳过此关系。请检查数据一致性。")
                continue

            # 清理实体类型以用于查询
            sanitized_head_type = Neo4jLoader._sanitize_entity_type(head_type)
            sanitized_tail_type = Neo4jLoader._sanitize_entity_type(tail_type)

            set_props = ", ".join([f"r.{k} = ${k}" for k in props])
            
            # 【核心修改】在MATCH语句中同时指定节点类型(Label)和名称(name)
            query = (
                f"MATCH (h:`{sanitized_head_type}` {{name: $head_name}}), (t:`{sanitized_tail_type}` {{name: $tail_name}}) "
                f"MERGE (h)-[r:`{rel_type}`]->(t) "
                "SET r.import_ids = CASE "
                "WHEN r.import_ids IS NULL THEN [$import_id] "
                "WHEN NOT $import_id IN r.import_ids THEN r.import_ids + $import_id "
                "ELSE r.import_ids END"
            )
            if set_props:
                query += f", {set_props}"
            
            params = {"head_name": head, "tail_name": tail, "import_id": import_id, **props}
            try:
                tx.run(query, **params)
            except exceptions.ClientError as e:
                # 提供了更详细的错误信息
                print(f"创建关系失败: ({head}:{sanitized_head_type}) -[{rel_type}]-> ({tail}:{sanitized_tail_type}) - {e}")
    def add_from_dicts(self, dicts, import_id):
        """批量添加多个条目（每个条目为一个dict，含entities和relations）"""
        for entry in dicts:
            entities = entry.get("entities", [])
            relations = entry.get("relations", [])
            if not entities and not relations:
                print("条目中没有找到实体或关系，跳过。")
                continue
            try:
                with self.driver.session(database=self.database) as session:
                    session.execute_write(
                        self._create_entities_and_relations,
                        entities,
                        relations,
                        import_id
                    )
                print(f"成功导入条目，导入ID: {import_id}")
            except exceptions.ServiceUnavailable as e:
                print(f"无法连接到 Neo4j 数据库。错误详情: {e}")
            except Exception as e:
                print(f"处理条目时发生错误: {e}")

    def remove_entities_and_relations_from_dicts(self, dicts, import_id):
        """批量移除多个条目中的实体和关系（只移除指定 import_id）"""
        for entry in dicts:
            entities = entry.get("entities", [])
            relations = entry.get("relations", [])
            entity_names = [e[0] for e in entities if len(e) == 2]
            rel_tuples = [tuple(r) for r in relations if len(r) == 3]
            self.remove_entities_and_relations(entity_names, rel_tuples, import_id)

    def clear_kg_by_import_id(self, import_id):
        """从所有节点和关系的 import_ids 列表中移除指定 import_id，若 import_ids 为空则删除节点/关系"""
        try:
            with self.driver.session(database=self.database) as session:
                # 处理关系
                session.run(
                    """
                    MATCH ()-[r]-()
                    WHERE $import_id IN r.import_ids
                    SET r.import_ids = [id IN r.import_ids WHERE id <> $import_id]
                    WITH r WHERE size(r.import_ids) = 0
                    DELETE r
                    """,
                    import_id=import_id
                )
                # 处理节点
                session.run(
                    """
                    MATCH (n)
                    WHERE $import_id IN n.import_ids
                    SET n.import_ids = [id IN n.import_ids WHERE id <> $import_id]
                    WITH n WHERE size(n.import_ids) = 0
                    DETACH DELETE n
                    """,
                    import_id=import_id
                )
                print(f"已移除 import_id 为 {import_id} 的所有节点和关系（如无其他 KG 引用则彻底删除）。")
        except Exception as e:
            print(f"删除知识图谱时发生错误: {e}")

    def remove_entities_and_relations(self, entity_names=None, relations=None, import_id=None):
        """
        从指定实体和关系的 import_ids 列表中移除 import_id，若 import_ids 为空则删除节点/关系
        entity_names: list of entity names to remove
        relations: list of (head, rel, tail) tuples to remove
        """
        try:
            with self.driver.session(database=self.database) as session:
                # 删除指定关系
                if relations:
                    for head, rel, tail in relations:
                        query = (
                            "MATCH (h {name: $head_name})-[r:`" + rel + "`]->(t {name: $tail_name}) "
                            "WHERE $import_id IN r.import_ids "
                            "SET r.import_ids = [id IN r.import_ids WHERE id <> $import_id] "
                            "WITH r WHERE size(r.import_ids) = 0 "
                            "DELETE r"
                        )
                        session.run(query, head_name=head, tail_name=tail, import_id=import_id)
                # 删除指定实体（及其所有关系）
                if entity_names:
                    for name in entity_names:
                        query = (
                            "MATCH (n {name: $name}) "
                            "WHERE $import_id IN n.import_ids "
                            "SET n.import_ids = [id IN n.import_ids WHERE id <> $import_id] "
                            "WITH n WHERE size(n.import_ids) = 0 "
                            "DETACH DELETE n"
                        )
                        session.run(query, name=name, import_id=import_id)
                print("指定实体和关系已移除 import_id（如无其他 KG 引用则彻底删除）。")
        except Exception as e:
            print(f"删除实体或关系时发生错误: {e}")

def load_dicts_from_file(data_file):
    """支持json或jsonl文件，返回dict列表"""
    dicts = []
    if data_file.endswith(".jsonl"):
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    dicts.append(json.loads(line))
    else:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                dicts = data
            elif isinstance(data, dict):
                dicts = [data]
            else:
                raise ValueError("不支持的数据格式")
    return dicts

NEO4JLOADER = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database=target_db_name)

# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--import-id", required=True, help="KG唯一标识")
#     parser.add_argument("--data_file", help="输入数据文件（json/jsonl）")
#     parser.add_argument("--action", required=True, choices=["add", "remove", "clear"], help="操作类型: add/remove/clear")
#     args = parser.parse_args()

#     target_db_name = "neo4j"
#     loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database=target_db_name)

#     if args.action == "add":
#         if not args.data_file:
#             print("add 操作需要 --data_file 参数")
#         else:
#             dicts = load_dicts_from_file(args.data_file)
#             loader.add_from_dicts(dicts, args.import_id)
#     elif args.action == "remove":
#         if not args.data_file:
#             print("remove 操作需要 --data_file 参数")
#         else:
#             dicts = load_dicts_from_file(args.data_file)
#             loader.remove_entities_and_relations_from_dicts(dicts, args.import_id)
#     elif args.action == "clear":
#         loader.clear_kg_by_import_id(args.import_id)
#     loader.close()

# if __name__ == "__main__":
#     main()

# --data_file test.json --import-id 9dda7d23-b772-4b3b-b9d3-25839b197b34 --action add
# --data_file remove.json --import-id 9dda7d23-b772-4b3b-b9d3-25839b197b34 --action remove
# --import-id 9dda7d23-b772-4b3b-b9d3-25839b197b34 --action clear

"""
retrieval command:
MATCH (n)-[r]->(m)
WHERE "9dda7d23-b772-4b3b-b9d3-25839b197b34" IN r.import_ids
RETURN n, r, m
"""