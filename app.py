import streamlit as st
import numpy as np
import pandas as pd
from numpy.linalg import svd, eigvals, eig

st.title('SIMPLE VECTOR MATRIX APPS')

with st.sidebar:
    tipe = st.radio('Pilih Tipe', ['single vector', 'double vector', 'single matrix', 'double matrix', 'Eigen', 'OBE', 'SVD', 'Quadratic Matrix'])

with st.expander('Pilih Ukuran'):
    with st.form('Pilih Ukuran'):
        if tipe == 'single vector':
            size = st.number_input('ukuran vektor', min_value=2)
        elif tipe == 'double matrix':
            row1 = st.number_input('ukuran baris matrix pertama', min_value=2)
            col1 = st.number_input('ukuran kolom matrix pertama', min_value=2)
            row2 = st.number_input('ukuran baris matrix kedua', min_value=2)
            col2 = st.number_input('ukuran kolom matrix kedua', min_value=2)
        elif tipe == 'double vector':
            size = st.number_input('ukuran double vector', min_value=2)
        submit = st.form_submit_button('submit size')

if tipe == 'single vector':
    df = pd.DataFrame(columns=range(1, size + 1), index=range(1, 2), dtype=float)
    st.write('Masukkan data untuk vektor')
    df_input = st.experimental_ui_editor(df, key='single_vector')

elif tipe == 'double vector':
    df = pd.DataFrame(columns=range(1, size + 1), index=range(1, 3), dtype=float)
    st.write('Masukkan data untuk double vector')
    df_input = st.experimental_ui_editor(df, key='double_vector')

elif tipe == 'single matrix':
    row = st.number_input('ukuran baris matrix', min_value=2)
    col = st.number_input('ukuran kolom matrix', min_value=2)
    df = pd.DataFrame(columns=range(1, col + 1), index=range(1, row + 1), dtype=float)
    st.write('Masukkan data untuk matrix')
    df_input = st.experimental_ui_editor(df, key='single_matrix')
    st.write('Matrix:')
    st.write(df_input)
    # Operasi atau manipulasi pada matrix
    # Tambahkan kode sesuai dengan kebutuhan Anda
    Operasi = st.radio('Pilih Operasi', ['A*B', 'A+B', 'Determinan', 'Invers'])
    matrix1 = df_input.fillna(0).to_numpy()
    if Operasi == 'A*B':
        # Lakukan operasi perkalian matriks dengan dirinya sendiri
        result = np.matmul(matrix1, matrix1)
        st.write(result)

    elif Operasi == 'A+B':
        # Lakukan operasi penjumlahan matriks dengan dirinya sendiri
        result = matrix1 + matrix1
        st.write(result)

    elif Operasi == 'Determinan':
        # Cari determinan matriks
        determinant = np.linalg.det(matrix1)
        st.write('Determinan:')
        st.write(determinant)

elif tipe == 'double matrix':
    df1 = pd.DataFrame(columns=range(1, col1 + 1), index=range(1, row1 + 1), dtype=float)
    st.write('Masukkan data untuk matrix pertama')
    df1_input = st.experimental_ui_editor(df1, key='matrix1')

    df2 = pd.DataFrame(columns=range(1, col2 + 1), index=range(1, row2 + 1), dtype=float)
    st.write('Masukkan data untuk matrix kedua')
    df2_input = st.experimental_ui_editor(df2, key='matrix2')

    Operasi = None  # Assign a default value
    Operasi = st.radio('Pilih Operasi', ['A*B', 'A+B', 'Determinan', 'Invers'])
    matrix1 = df1_input.fillna(0).to_numpy()
    matrix2 = df2_input.fillna(0).to_numpy()

    if Operasi == 'A*B':
        # Lakukan operasi perkalian matriks
        result = np.matmul(matrix1, matrix2)
        st.write(result)

    elif Operasi == 'A+B':
        # Lakukan operasi penjumlahan matriks
        result = matrix1 + matrix2
        st.write(result)

    elif Operasi == 'Determinan':
        # Cari determinan matriks pertama
        determinant = np.linalg.det(matrix1)
        st.write('Determinan matrix pertama:')
        st.write(determinant)

        # Cari determinan matriks kedua
        determinant = np.linalg.det(matrix2)
        st.write('Determinan matrix kedua:')
        st.write(determinant)

    elif Operasi == 'Invers':
        # Cari invers matriks pertama
        inverse = np.linalg.inv(matrix1)
        st.write('Invers matrix pertama:')
        st.write(inverse)

        # Cari invers matriks kedua
        inverse = np.linalg.inv(matrix2)
        st.write('Invers matrix kedua:')
        st.write(inverse)

elif tipe == 'Eigen':
    df = pd.DataFrame(columns=range(1, col + 1), index=range(1, row + 1), dtype=float)
    st.write('Masukkan data untuk matrix')
    df_input = st.experimental_ui_editor(df, key='eigen_matrix')
    matrix = df_input.fillna(0).to_numpy()
    eigenvalues = eigvals(matrix)
    st.write('Eigenvalues:')
    st.write(eigenvalues)

elif tipe == 'OBE':
    df = pd.DataFrame(columns=range(1, col + 1), index=range(1, row + 1), dtype=float)
    st.write('Masukkan data untuk matrix')
    df_input = st.experimental_ui_editor(df, key='obe_matrix')
    matrix = df_input.fillna(0).to_numpy()
    rref = np.linalg.matrix_rank(matrix)
    st.write('Reduced Row Echelon Form:')
    st.write(rref)

elif tipe == 'SVD':
    df = pd.DataFrame(columns=range(1, col + 1), index=range(1, row + 1), dtype=float)
    st.write('Masukkan data untuk matrix')
    df_input = st.experimental_ui_editor(df, key='svd_matrix')
    matrix = df_input.fillna(0).to_numpy()
    U, s, V = np.linalg.svd(matrix)
    st.write('U:')
    st.write(U)
    st.write('s:')
    st.write(s)
    st.write('V:')
    st.write(V)

elif tipe == 'Quadratic Matrix':
    n = st.number_input('ukuran matrix', min_value=2)
    df = pd.DataFrame(columns=range(1, n + 1), index=range(1, n + 1), dtype=float)
    st.write('Masukkan data untuk matrix')
    df_input = st.experimental_ui_editor(df, key='quadratic_matrix')
    matrix = df_input.fillna(0).to_numpy()
    is_positive_definite = np.all(np.linalg.eigvals(matrix) > 0)
    st.write('Is Positive Definite:')
    st.write(is_positive_definite)
