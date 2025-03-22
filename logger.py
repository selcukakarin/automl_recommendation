import logging
import os
import sys
from datetime import datetime
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Log seviyeleri
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

class CustomFormatter(logging.Formatter):
    """Özel log formatını tanımlayan sınıf"""
    
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: BLUE + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatında log çıktısı veren sınıf"""
    
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record, ensure_ascii=False)

def get_log_file_path(name, log_dir, log_type="general"):
    """Log dosyası yolunu oluşturur"""
    # Ana log dizini
    base_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m'))
    
    # Log tipi için alt dizin
    type_dir = os.path.join(base_dir, log_type)
    os.makedirs(type_dir, exist_ok=True)
    
    # Log dosya adı
    if name:
        filename = f"{name}.log"
    else:
        filename = "general.log"
    
    return os.path.join(type_dir, filename)

def setup_logger(name=None, level="info", log_file=None, console_output=True, 
                rotation="size", max_bytes=10*1024*1024, backup_count=5, json_output=False):
    """
    Log sistemini ayarlar
    
    Parametreler:
        name (str): Logger adı, None ise root logger kullanılır
        level (str): Log seviyesi ('debug', 'info', 'warning', 'error', 'critical')
        log_file (str): Log dosyasının tam yolu
        console_output (bool): Konsola da çıktı verilecek mi
        rotation (str): 'size' veya 'time' olarak dosya rotasyon tipi
        max_bytes (int): Rotasyon için maksimum dosya boyutu (bytes)
        backup_count (int): Tutulacak eski log dosyası sayısı
        json_output (bool): JSON formatında log çıkarmak için
    
    Returns:
        logging.Logger: Ayarlanmış logger nesnesi
    """
    
    # Logger'ı oluştur
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
    
    # Eğer handler'lar eklenmiş ise, temizle
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Log dosya yolunu kontrol et
    if log_file is None:
        log_file = "app.log"  # Varsayılan log dosyası
    
    # Log dosyasının dizinini oluştur
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Dosya formatını ayarla
    file_formatter = JSONFormatter() if json_output else logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Rotasyon tipi seçimi
    if rotation.lower() == "time":
        # Günlük rotasyon
        file_handler = TimedRotatingFileHandler(
            log_file, when='midnight', interval=1, backupCount=backup_count, encoding='utf-8'
        )
    else:
        # Boyut bazlı rotasyon (varsayılan)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
        )
    
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Konsol çıktısı ekle
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)
    
    return logger

# Test performans metriklerini kaydetmek için özel logger fonksiyonu
def log_performance_metrics(logger, metrics, test_name, run_id=None):
    """
    Test performans metriklerini yapılandırılmış bir şekilde loglar
    
    Parametreler:
        logger (logging.Logger): Kullanılacak logger nesnesi
        metrics (dict): Performans metrikleri sözlüğü
        test_name (str): Test adı
        run_id (str): Test çalıştırma ID'si (None ise otomatik oluşturulur)
    """
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logger.info(f"--- Performance Metrics: {test_name} (Run ID: {run_id}) ---")
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"{metric_name}: {metric_value:.6f}")
        else:
            logger.info(f"{metric_name}: {metric_value}")
    
    logger.info("-" * 50)

# API istekleri için özel logger fonksiyonu
def log_api_request(logger, method, url, data=None, response=None, error=None, duration=None):
    """
    API isteklerini ve yanıtlarını loglar
    
    Parametreler:
        logger (logging.Logger): Kullanılacak logger nesnesi
        method (str): HTTP metodu (GET, POST, etc.)
        url (str): İstek URL'i
        data (dict): İstek verileri
        response (dict/str): API yanıtı
        error (Exception): Oluşan hata
        duration (float): İstek süresi (saniye)
    """
    log_data = {
        "method": method,
        "url": url
    }
    
    if data:
        log_data["request_data"] = data
        
    if response:
        log_data["response"] = response
        
    if error:
        log_data["error"] = str(error)
        
    if duration:
        log_data["duration"] = f"{duration:.4f}s"
    
    message = f"API Request: {method} {url}"
    if duration:
        message += f" ({duration:.4f}s)"
    
    if error:
        logger.error(message)
        logger.error(f"Request details: {json.dumps(log_data, ensure_ascii=False)}")
    else:
        logger.info(message)
        logger.debug(f"Request details: {json.dumps(log_data, ensure_ascii=False)}")

# Test başlangıç ve bitiş logları için yardımcı fonksiyonlar
def log_test_start(logger, test_name, parameters=None):
    """Test başlangıcını loglar"""
    logger.info(f"===== STARTING TEST: {test_name} =====")
    if parameters:
        logger.info(f"Test parameters: {json.dumps(parameters, ensure_ascii=False)}")

def log_test_end(logger, test_name, success=True, duration=None, summary=None):
    """Test bitişini loglar"""
    status = "SUCCEEDED" if success else "FAILED"
    message = f"===== TEST {status}: {test_name} ====="
    
    if duration:
        message += f" (Duration: {duration:.2f}s)"
    
    if success:
        logger.info(message)
    else:
        logger.error(message)
    
    if summary:
        logger.info(f"Test summary: {summary}") 