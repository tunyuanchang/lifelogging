# Lifelogging Server
###### cr: tunyuanchang

### Prerequisites
- FFMPEG
- MongoDB _(by default, the settings look for MongoDB running on "mongodb://localhost:27017". If that's not correct, change it in the combine_analysis_files.py, the create_texts_mongodb.py, and the local-config files)_

### Installation
_Optional - Create venv and activate it_

```bash
conda create -n divexplore python=3.9
conda activate divexplore
```

Required dependencies: 

```bash
cd divexplore
pip install -r requirements.txt
```

### Usage
1. Video Process

```bash
cd backend/analysis
bash new_process_videos.sh [path_to_video]
```

2. Frame Selection

```bash
cd backend
bash new_selection.sh
```

3. Database Integration

```bash
cd backend
bash new_integration.sh
```

4. Start FAISS Index Server

```bash
cd middleware/FAISS
bash start_clipserver.sh
```

5. Query VLM

```bash
cd frontend
bash start_queryserver.sh
```
