import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DBLogger:
    '''
    This class is used to log the database build process.
    '''
    def __init__(self, log_file: str = "db_build_log.json"):
        self.log_file = log_file
        self.current_log = self._load_log()
    
    def _load_log(self) -> Dict:
        '''
        This method is used to load the existing log file.
        '''
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading log file: {e}")
                return self._create_new_log()
        return self._create_new_log()
    
    def _create_new_log(self) -> Dict:
        '''
        This method is used to create a new log structure.
        '''
        return {
            "last_update": "",
            "db_path": "",
            "index_type": "",
            "index_config": {},
            "files": {},
            "total_documents": 0
        }
    
    def update_index_info(self, index_type: str, index_config: Dict):
        '''
        This method is used to update the index configuration information.
        '''
        self.current_log["index_type"] = index_type
        self.current_log["index_config"] = index_config
        
    def _calculate_file_hash(self, file_path: str) -> str:
        '''
        This method is used to calculate the MD5 hash value of the file.
        '''
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_file_info(self, file_path: str) -> Dict:
        '''
        This method is used to get the file information.
        '''
        return {
            "hash": self._calculate_file_hash(file_path),
            "size": os.path.getsize(file_path),
            "last_modified": datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).strftime('%Y-%m-%d %H:%M:%S'),
            "doc_count": 0  # 将在处理文件时更新
        }
    
    def check_files_changed(self, data_folder: str, file_pattern: str = "*.jsonl") -> List[str]:
        '''
        This method is used to check if the file has changed, and return the list of files that need to be updated.
        '''
        import glob
        
        changed_files = []
        current_files = set()
        
        # 检查现有文件
        for file_path in glob.glob(os.path.join(data_folder, file_pattern)):
            current_files.add(file_path)
            file_info = self._get_file_info(file_path)
            
            if file_path not in self.current_log["files"]:
                # 新文件
                changed_files.append(file_path)
                logger.info(f"New file detected: {file_path}")
            elif self.current_log["files"][file_path]["hash"] != file_info["hash"]:
                # 文件已修改
                changed_files.append(file_path)
                logger.info(f"File changed: {file_path}")
        
        # 检查删除的文件
        for old_file in self.current_log["files"].keys():
            if old_file not in current_files:
                logger.warning(f"File removed: {old_file}")
        
        return changed_files
    
    def update_file_record(self, file_path: str, doc_count: int):
        """更新文件记录"""
        file_info = self._get_file_info(file_path)
        file_info["doc_count"] = doc_count
        self.current_log["files"][file_path] = file_info
    
    def save_db_build_log(self, db_path: str, total_documents: int):
        """保存数据库构建日志"""
        self.current_log["last_update"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.current_log["db_path"] = './logs/'
        self.current_log["total_documents"] = total_documents
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_log, f, indent=2, ensure_ascii=False)
        
        logger.info(f"DB build log updated: {self.log_file}")