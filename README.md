# DR-Vision: Diabetic Retinopathy Detection from Fundus Images

## Project Overview

**DR-Vision** is a deep learning project for **automated detection of Diabetic Retinopathy (DR)** using retinal fundus images.  
The project integrates multiple datasets from different sources, applies preprocessing, data augmentation, and exploratory data analysis (EDA), and prepares the data for training deep learning models such as **Swin Transformer**.  

Key objectives of the project:
- Build a robust unified dataset of fundus images labeled for DR detection.
- Enhance dataset quality through preprocessing and augmentation.
- Perform EDA to understand data distribution, image properties, and pixel statistics.
- Enable training and evaluation of state-of-the-art models for DR classification.
- Provide a dashboard for visualizing predictions and exploring results.

---

## Data Sources

### 1. Paraguay Fundus Dataset
- **Source:** Zenodo  
- **Content:** Raw fundus images labeled for DR, including both healthy and affected cases.  
- **Link:** [Paraguay Fundus Dataset](https://zenodo.org/records/4532361)

### 2. High-Resolution Fundus (HRF) Database
- **Source:** FAU Erlangen/Nürnberg  
- **Content:** Raw fundus images covering healthy, DR, and glaucoma patients.  
- **Link:** [HRF Database](https://lme.tf.fau.de/)

### 3. Bangladesh Multi-Disease Fundus Dataset
- **Source:** Scientific Data (Elsevier)  
- **Content:** Multi-disease retinal images including DR, glaucoma, and macular degeneration.  
- **Link:** [Bangladesh Fundus Dataset](https://www.sciencedirect.com/science/article/pii/S2352340924009417)

---

## Project Workflow / What We Have Done

1. **Dataset Collection and Organization**
   - Downloaded datasets from multiple sources.
   - Organized data into a unified structure: `train`, `val`, and `test` splits with `DR` and `No_DR` classes.

2. **Preprocessing**
   - Applied circle cropping to focus on the retina.
   - Used CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement.
   - Resized images to a standard input size (224x224 or 512x512).
   
3. **Data Augmentation**
   - Random rotations, horizontal & vertical flips.
   - Autocontrast, equalization, and solarization.
   - Generated 30 augmented images per original image (for HRF dataset).

4. **Exploratory Data Analysis (EDA)**
   - Class distribution across all splits.
   - Random sample image visualization.
   - Image size and aspect ratio distributions.
   - Approximate RGB channel mean & standard deviation.

5. **Unified Dataset Preparation**
   - Merged original and augmented images into `Unified_Split`.
   - Ensured `train` contains both original and augmented images, `val` and `test` only contain original images.

6. **Model Training**
   - Prepared data loaders and preprocessing pipeline for model input.
   - Trained classification models (e.g., Swin Transformer, ResNet variants).

7. **Dashboard**
   - Interactive interface to visualize predictions on sample images.
   - Display metrics and class probabilities for uploaded images.

---

## Create and Activate the Virtual Environment

It's highly recommended to use a **virtual environment** (`venv`) to manage the project dependencies and prevent conflicts with other Python projects.

1.  **Create the environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the environment:**
    ```bash
    # For Windows:
    venv\Scripts\activate

    # For Linux/macOS:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Install all required packages listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

---

## Run the Streamlit Dashboard

Once the environment is active and all dependencies are installed, you can launch the application.

```bash
streamlit run app.py

```

### Data Sets

The datasets and model weights are available here:
https://drive.google.com/drive/folders/1ynjAfxWf3_bycjVLyOi5AHZlEqbohKlO?usp=sharing



