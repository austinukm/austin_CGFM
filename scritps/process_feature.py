### this scritps is aim to procee the feature data 
import pandas as pd
import torch
import pathlib
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import scritps.project_config as project_config



def preprocess_text_feature_optimized(movie_data_csv, plots_dir, out_put_dir):
    # --- 1. 数据加载与准备 ---
    out_put_dir_path = pathlib.Path(out_put_dir)
    movies = pd.read_csv(
        movie_data_csv,
        sep="::",
        names=["movieId", "title", "cleaned_title", "genres"],
        encoding="latin-1",
        engine="python"
    )
    print("Loaded movie data with shape:", movies.shape)

    # --- 2. 模型初始化 ---
    model_name = "all-MiniLM-L6-v2"
    text_embedding_model = SentenceTransformer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_embedding_model.to(device)
    print(f"Using {model_name} on {device} for text encoding.")

    # 用于收集待编码的文本和对应的ID
    texts_to_encode = []
    movie_ids_with_plot = []
    
    # --- 3. 数据收集 (代替逐个编码) ---
    print("Collecting and preparing texts...")
    with open(out_put_dir_path / "miss_plots.log", "w",encoding='utf-8') as log_file:
        for indx, row in movies.iterrows():
            movie_id = row["movieId"]
            cleaned_title = row["cleaned_title"]
            genres = row["genres"]
            title = row["title"] # Keep title for logging

            # 假设 plot 文件名后缀是 .txt
            plot_file_path = pathlib.Path(f"{plots_dir}/{movie_id}.txt")
            
            if not plot_file_path.exists():
                log_file.write(f"Missing plot for movieId: {movie_id}, title: {title} {plot_file_path}\n")
            else:
                description = plot_file_path.read_text(encoding='utf-8', errors='ignore') # 推荐指定编码
                text = f"{cleaned_title}. {genres}. {description}"
                
                texts_to_encode.append(text)
                movie_ids_with_plot.append(movie_id)

    # --- 4. 批量编码 (核心优化) ---
    print(f"Starting batch encoding of {len(texts_to_encode)} texts...")
    
    # 设置 batch_size，通常 32, 64 或 128 是个好的起点
    # show_progress_bar=True 可以查看进度
    batch_size = 64 
    
    all_embeddings = text_embedding_model.encode(
        texts_to_encode, 
        convert_to_tensor=True,
        device=device,
        batch_size=batch_size, 
        show_progress_bar=True
    ).cpu().numpy() # 编码完成后移回 CPU 并转为 numpy 数组

    # --- 5. 存储结果 ---
    
    # 将 ID 和特征转换为 list/numpy 格式以便存储
    ids = movie_ids_with_plot
    # all_embeddings 已经是 numpy 数组，可以直接存储

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = out_put_dir_path / f"movie_text_features_{current_time}.npz"
    
    np.savez_compressed(
        file=output_filename,
        ids=ids,
        features=all_embeddings
    )
    
    print(f"Text features saved successfully to {output_filename}. Length: {len(ids)}.")
    
    # 额外创建字典用于后续查找（可选）
    text_feature_dict = dict(zip(ids, all_embeddings))
    return text_feature_dict

    



def preprocess_img_feature(movie_data_csv,img_dir,out_put_dir):
    out_put_dir_path  = pathlib.Path(out_put_dir)
    movie = pd.read_csv(
        movie_data_csv,
        sep = "::",
        names = ["movieId", "title", "cleaned_title", "genres"],
        encoding="latin-1",
        engine="python"
    )
    print("loaded movie data with shape:", movie.shape)
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    img_model = CLIPModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_model.to(device)
    img_model.eval()
    print(f"Using {model_name} for image feature extraction on {device}. dimmension : {img_model.config.projection_dim}")
    image_list = []
    movie_ids_with_poster = []
    print("Collecting and validating poster files...")
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    with open(out_put_dir_path / f"miss_plots_{current_time}.log","w") as log_file:
        for indx, row in movie.iterrows():
            movie_id = row["movieId"]
            poster_file_path = pathlib.Path(f"{img_dir}/{movie_id}.jpg")
            if not poster_file_path.exists():
                log_file.write(f"Missing poster for movieId: {movie_id}\n")                
            else :
                try :
                    image = Image.open(poster_file_path).convert("RGB")
                    image_list.append(image)
                    movie_ids_with_poster.append(movie_id)
                except Exception as e:
                    log_file.write(f"Error loading image for movieId: {movie_id}, error: {e}\n")
    BATCH_SIZE = 64
    img_feature_dict = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(image_list), BATCH_SIZE)):
            batch_images = image_list[i:i + BATCH_SIZE]
            batch_movie_ids = movie_ids_with_poster[i:i + BATCH_SIZE]
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            image_features = img_model.get_image_features(pixel_values=inputs.pixel_values)
            image_features_normalized = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            features_np = image_features_normalized.cpu().numpy()
            for idx, movie_id in enumerate(batch_movie_ids):
                img_feature_dict[movie_id] = features_np[idx].squeeze()
    ids = list(img_feature_dict.keys())
    features = list(img_feature_dict.values())
    out_put_npz_path = out_put_dir_path / f"movie_img_features_{current_time}.npz"
    np.savez_compressed(
        file=out_put_npz_path,
        ids=np.array(ids),
        features=np.stack(features)
    )
    print(f"\nImage features saved successfully to {out_put_npz_path}. Length: {len(ids)}.")
    return img_feature_dict




plots_dir = r"D:\work_space\work\austin_paper_coding\austin_paper1\data\primary_all\plots"

covers_dir = r"D:\work_space\work\austin_paper_coding\austin_paper1\data\primary_all\covers"

movie_data_csv = r"D:\work_space\work\austin_paper_coding\austin_paper1\data\movies_with_clean_titles.dat"

out_put_dir = r"D:\work_space\work\austin_paper_coding\austin_paper1\feature_output"


preprocess_text_feature_optimized(movie_data_csv, plots_dir, out_put_dir)

preprocess_img_feature(movie_data_csv, covers_dir, out_put_dir)

 