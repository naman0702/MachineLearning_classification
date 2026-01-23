{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "505d2152-b8fb-4c9a-a00e-7309f050e04b",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax. Perhaps you forgot a comma? (1073483546.py, line 50)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[1], line 50\u001b[1;36m\u001b[0m\n\u001b[1;33m    [\u001b[0m\n\u001b[1;37m    ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax. Perhaps you forgot a comma?\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "st.set_page_config(page_title=\"ML Model Evaluation App\", layout=\"wide\")\n",
    "st.title(\"📊 Machine Learning Model Evaluation Dashboard\")\n",
    "\n",
    "st.markdown(\n",
    "    \"\"\"\n",
    "    Upload **test data only (CSV)** and evaluate different classification models.\n",
    "    \"\"\"\n",
    ")\n",
    "\n",
    "# (a) Dataset upload (CSV)\n",
    "# --------------------------------------------------\n",
    "uploaded_file = st.file_uploader(\n",
    "    \"📂 Upload Test Dataset (CSV)\",\n",
    "    type=[\"csv\"]\n",
    ")\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    df = pd.read_csv(uploaded_file)\n",
    "    st.success(\"Dataset uploaded successfully!\")\n",
    "\n",
    "    st.subheader(\"🔍 Dataset Preview\")\n",
    "    st.dataframe(df.head(), use_container_width=True)\n",
    "\n",
    "    # --------------------------------------------------\n",
    "    # Assume last column is target\n",
    "    # --------------------------------------------------\n",
    "    X = df.iloc[:, :-1]\n",
    "    y = df.iloc[:, -1]\n",
    "\n",
    "    # --------------------------------------------------\n",
    "    # Train-test split (only for demo)\n",
    "    # --------------------------------------------------\n",
    "    X_train, X_test, y_train, y_test = train_test_split(\n",
    "        X, y, test_size=0.3, random_state=42, stratify=y\n",
    "    )\n",
    "\n",
    "    # --------------------------------------------------\n",
    "    # (b) Model selection dropdown\n",
    "    # --------------------------------------------------\n",
    "    model_name = st.selectbox(\n",
    "        \"🤖 Select Classification Model\",\n",
    "        [\n",
    "            \"Logistic Regression\",\n",
    "            \"Decision Tree\",\n",
    "            \"K-Nearest Neighbors\"\n",
    "        ]\n",
    "    )\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12e5d6da-7217-4fde-abb2-a1b5ee0519e5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
