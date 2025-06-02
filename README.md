安裝套件
pip install -r requirements.txt

如果有TensorFlow DLL 載入失敗的錯誤：
有時安裝過程損壞或有殘留，請完整移除再安裝一次：
pip uninstall tensorflow keras -y
pip cache purge
pip install tensorflow

執行
python LLM.py