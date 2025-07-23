import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from skimage.transform import rotate
import argparse
import os

def detectar_angulo_projecao_horizontal(imagem):
    _, binaria = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    melhor_angulo = 0
    melhor_valor = -1
    angulos_testados = np.arange(-90, 90, 0.5)
    variacoes= []

    for angulo in angulos_testados:
        rotacionada = rotate(binaria, angulo, resize=True, mode='edge', preserve_range=True).astype(np.uint8)
        perfil = np.sum(rotacionada, axis=1)
        variacao = np.sum(np.diff(perfil) ** 2)
        variacoes.append(variacao)

        if variacao > melhor_valor:
            melhor_valor = variacao
            melhor_angulo = angulo

    return melhor_angulo

def rotacionar(imagem, angulo):
    (h, w) = imagem.shape[:2]
    centro = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(imagem, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def detectar_angulo_hough_melhorado(imagem, exibir_resultado=False):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    equalizada = clahe.apply(cinza)
    suave = cv2.GaussianBlur(equalizada, (5, 5), 0)

    v = np.median(suave)
    sigma = 0.33
    limiar_inferior = int(max(0, (1.0 - sigma) * v))
    limiar_superior = int(min(255, (1.0 + sigma) * v))
    bordas = cv2.Canny(suave, limiar_inferior, limiar_superior)

    linhas = cv2.HoughLines(bordas, 1, np.pi / 180, threshold=250)
    if linhas is None:
        linhas = cv2.HoughLines(bordas, 1, np.pi / 180, threshold=45)
        if linhas is None:
            print("    [!] Nenhuma linha detectada (mesmo com threshold reduzido).")
            return None, imagem

    angulos = []
    imagem_resultado = imagem.copy()
    for linha in linhas:
        rho, theta = linha[0]
        angulo = np.rad2deg(theta) - 90
        angulos.append(angulo)

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(imagem_resultado, (x1, y1), (x2, y2), (0, 0, 255), 1)

    theta_M = np.median(angulos)

    (h, w) = imagem.shape[:2]
    centro = (w // 2, h // 2)
    matriz_rotacao = cv2.getRotationMatrix2D(centro, theta_M, 1.0)
    imagem_corrigida = cv2.warpAffine(imagem, matriz_rotacao, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if exibir_resultado:
        imagem_rgb = cv2.cvtColor(imagem_resultado, cv2.COLOR_BGR2RGB)
        imagem_corrigida_vis = cv2.cvtColor(imagem_corrigida, cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[0].imshow(imagem_rgb)
        axs[0].set_title(f"Linhas detectadas (θₘ = {theta_M:.2f}°)")
        axs[0].axis('off')

        axs[1].hist(angulos, bins=90, range=(-90, 90), color='gray', edgecolor='black')
        axs[1].axvline(theta_M, color='red', linestyle='--', label=f'θₘ = {theta_M:.2f}°')
        axs[1].set_title("Histograma dos ângulos (Hough)")
        axs[1].set_xlabel("Ângulo (graus)")
        axs[1].set_ylabel("Contagem")
        axs[1].legend()

        axs[2].imshow(imagem_corrigida_vis)
        axs[2].set_title("Imagem corrigida")
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()

    return theta_M, imagem_corrigida

def avaliar_ocr(imagem):
    if len(imagem.shape) == 3:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    texto = pytesseract.image_to_string(imagem)
    num_caracteres = len(texto)
    return texto, num_caracteres

def main():
    parser = argparse.ArgumentParser(description="Corrigir rotação de imagem usando projeção ou Hough.")
    parser.add_argument("imagem", type=str, help="Caminho da imagem")
    parser.add_argument("modo", choices=["projecao", "hough"], help="Modo de correção (projecao ou hough)")
    args = parser.parse_args()

    caminho = args.imagem
    modo = args.modo

    if not os.path.exists(caminho):
        print(f"[!] Caminho inválido: {caminho}")
        return
    
    if modo == "projecao":
        imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        angulo = detectar_angulo_projecao_horizontal(imagem)
        print(f"[{caminho}] Ângulo detectado: {angulo:.2f} graus")
        imagem_corrigida = rotacionar(imagem, angulo)

        # Gerar função objetivo novamente para plot
        _, binaria = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        angulos_testados = np.arange(-90, 90, 0.5)
        variacoes = []
        for a in angulos_testados:
            rotacionada = rotate(binaria, a, resize=True, mode='edge', preserve_range=True).astype(np.uint8)
            perfil = np.sum(rotacionada, axis=1)
            variacao = np.sum(np.diff(perfil) ** 2)
            variacoes.append(variacao)

        # Exibição conjunta
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Gráfico da função objetivo
        axes[0].plot(angulos_testados, variacoes)
        axes[0].axvline(angulo, color='red', linestyle='--', label=f'{angulo:.2f}°')
        axes[0].set_title("Função objetivo (perfil horizontal)")
        axes[0].set_xlabel("Ângulo (graus)")
        axes[0].set_ylabel("Variação")
        axes[0].grid(True)
        axes[0].legend()

        # Imagem original
        axes[1].imshow(imagem, cmap='gray')
        axes[1].set_title("Imagem original")
        axes[1].axis('off')

        # Imagem corrigida
        axes[2].imshow(imagem_corrigida, cmap='gray')
        axes[2].set_title(f"Imagem corrigida ({angulo:.2f}°)")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    elif modo == "hough":
        imagem = cv2.imread(caminho)
        angulo, imagem_corrigida = detectar_angulo_hough_melhorado(imagem, exibir_resultado=True)
        if angulo is not None:
            print(f"[{caminho}] Ângulo estimado: {angulo:.2f} graus")
        else:
            print("    Nenhuma linha detectada.")

    # Avaliação OCR
    print("\n[OCR] Texto antes do alinhamento:")
    texto_antes, num_antes = avaliar_ocr(imagem)
    print(texto_antes)
    print(f"--> Número de caracteres detectados antes: {num_antes}")

    print("\n[OCR] Texto após o alinhamento:")
    texto_depois, num_depois = avaliar_ocr(imagem_corrigida)
    print(texto_depois)
    print(f"--> Número de caracteres detectados depois: {num_depois}")

if __name__ == "__main__":
    main()

