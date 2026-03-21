import pytest
import torch
import os
import logging
from pathlib import Path
import json
import shutil
from contextlib import contextmanager
from data_generators.data_generator import DataGenerator
from trainers.trainer import Trainer
from utils import set_seed
from configs.sid4srec_config import get_config

class TestSID4SRec:
    @pytest.fixture(scope="class")
    def base_config(self):
        """提供基礎配置"""
        return get_config()
    
    @pytest.fixture(scope="class")
    def device(self):
        """設置設備"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """設置和清理測試環境"""
        # 設置測試目錄
        test_dir = Path("test_outputs")
        test_dir.mkdir(exist_ok=True)
        
        # 設置日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(test_dir / "test.log"),
                logging.StreamHandler()
            ]
        )
        
        yield
        
        # 清理測試目錄
        if test_dir.exists():
            shutil.rmtree(test_dir)
    
    @contextmanager
    def setup_test_env(self, base_config, param_dict):
        """設置單個測試環境的上下文管理器"""
        # 備份原始配置
        original_attrs = {key: getattr(base_config, key) for key in param_dict.keys()}
        
        try:
            # 更新配置
            for key, value in param_dict.items():
                setattr(base_config, key, value)
            
            # 設置測試專用目錄
            test_name = "_".join(f"{k}-{v}" for k, v in param_dict.items())
            test_dir = Path("test_outputs") / test_name
            test_dir.mkdir(parents=True, exist_ok=True)
            
            base_config.output_dir = str(test_dir)
            base_config.checkpoint_path = str(test_dir / "model.pt")
            base_config.log_file = str(test_dir / "test.log")
            
            yield base_config
            
        finally:
            # 恢復原始配置
            for key, value in original_attrs.items():
                setattr(base_config, key, value)
    
    def run_single_test(self, config, device, param_dict):
        """執行單個測試配置"""
        try:
            logging.info(f"開始測試參數: {param_dict}")
            
            set_seed(config.seed)
            data_generator = DataGenerator(config)
            trainer = Trainer(config, device, data_generator)
            
            # 執行訓練
            trainer.train()
            
            # 記錄成功結果
            results = {
                "parameters": param_dict,
                "status": "success"
            }
            
            results_file = Path(config.output_dir) / "results.json"
            results_file.write_text(json.dumps(results, indent=4))
            
            logging.info(f"測試成功完成: {param_dict}")
            return True
            
        except Exception as e:
            logging.error(f"測試失敗 - 參數: {param_dict}, 錯誤: {str(e)}")
            
            # 記錄失敗結果
            results = {
                "parameters": param_dict,
                "error": str(e),
                "status": "failed"
            }
            
            results_file = Path(config.output_dir) / "results.json"
            results_file.write_text(json.dumps(results, indent=4))
            
            return False
    
    @pytest.mark.parametrize("dataset", [
        "Beauty",
        "Sports_and_Outdoors",
        "Home_and_Kitchen",
        "Toys_and_Games"
    ])
    @pytest.mark.parametrize("params", [
        {"alpha": 0.1, "beta": 0.1, "gamma": 0.1},
        {"alpha": 0.3, "beta": 0.3, "gamma": 0.3},
        {"alpha": 0.5, "beta": 0.5, "gamma": 0.5},
        {"alpha": 0.7, "beta": 0.7, "gamma": 0.7},
        {"alpha": 0.9, "beta": 0.9, "gamma": 0.9}
    ])
    def test_model_parameters(self, base_config, device, dataset, params):
        """測試不同數據集和模型參數組合"""
        test_params = {"dataset": dataset, **params}
        
        with self.setup_test_env(base_config, test_params) as test_config:
            assert self.run_single_test(test_config, device, test_params)
    
    @pytest.mark.parametrize("dataset", ["Beauty"])  # 使用較小的數據集進行快速測試
    @pytest.mark.parametrize("training_params", [
        {"mlm_probability_train": 0.1, "phi": 0.1, "lambda_cl": 0.1},
        {"mlm_probability_train": 0.5, "phi": 0.5, "lambda_cl": 0.5},
        {"mlm_probability_train": 0.9, "phi": 0.9, "lambda_cl": 0.9}
    ])
    def test_training_parameters(self, base_config, device, dataset, training_params):
        """測試不同訓練參數組合"""
        test_params = {"dataset": dataset, **training_params}
        
        with self.setup_test_env(base_config, test_params) as test_config:
            assert self.run_single_test(test_config, device, test_params)
