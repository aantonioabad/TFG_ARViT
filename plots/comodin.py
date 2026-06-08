import os
import re
import json
import pandas as pd
import numpy as np

# 1. Definimos los valores de energía exacta (E0) de tu tabla anterior
# El diccionario usa la tupla (alpha, J) como clave
E_exacta_dict = {
    (1.0, -2.0): -44.2395,
    (2.5, -4.0): -51.7333,
    (6.0, -4.0): -41.3075,
    (1.0,  7.0): -48.3624,
    (2.5,  2.75): -24.8600,
    (6.0,  7.0): -69.3503
}

# Número de espines N (según tu tabla anterior es una red 4x4)
N_spins = 16 

def parse_netket_log(filepath):
    """
    Lee el archivo de log y extrae la Energía Media y la Varianza del último paso.
    Asume formato JSON típico de NetKet.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # En NetKet, normalmente se guarda dentro de 'Energy' en cada iteración
        # Cogemos el último valor de la optimización
        if 'Energy' in data:
            last_energy = data['Energy']['Mean'][-1]
            last_variance = data['Energy']['Variance'][-1]
        else:
            # Si el formato JSON es una lista de diccionarios
            last_step = data[-1]
            last_energy = last_step['Energy']['Mean']
            last_variance = last_step['Energy']['Variance']
            
        return np.real(last_energy), np.real(last_variance)
    
    except json.JSONDecodeError:
        # Fallback por si el log es de texto plano y no JSON
        last_energy, last_variance = None, None
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                # Busca patrones comunes de texto, ajusta si es necesario
                if "Energy" in line or "Mean" in line:
                    match_e = re.search(r'Mean[:=]\s*([-+]?\d*\.\d+)', line)
                    match_v = re.search(r'Variance[:=]\s*([-+]?\d*\.\d+)', line)
                    if match_e and match_v:
                        last_energy = float(match_e.group(1))
                        last_variance = float(match_v.group(1))
                        break
        return last_energy, last_variance

def analizar_logs(directorio="."):
    resultados = []
    
    # Expresión regular para extraer información del nombre del archivo
    # Ej: resultado_direct_alpha1.0_J-2.0.log
    patron_archivo = re.compile(r'resultado_(direct|metropolis)_alpha([\d.]+)_J([-.\d]+)\.log')
    
    for filename in os.listdir(directorio):
        match = patron_archivo.match(filename)
        if match:
            muestreo = match.group(1).capitalize()
            alpha = float(match.group(2))
            J = float(match.group(3))
            
            # Obtener E0 para este alfa y J
            E0 = E_exacta_dict.get((alpha, J))
            if E0 is None:
                print(f"⚠️ No hay E0 definido para alpha={alpha}, J={J}. Saltando {filename}")
                continue
                
            # Extraer energía y varianza del archivo
            filepath = os.path.join(directorio, filename)
            E_VQMC, Varianza = parse_netket_log(filepath)
            
            if E_VQMC is not None and Varianza is not None:
                # Calcular Error Relativo
                E_rel = abs((E_VQMC - E0) / E0)
                
                # Calcular V-Score
                # Fórmula estándar en la literatura VMC: V-Score = N * Var(E) / (E_VQMC - E_exacta)^2
                delta_E = abs(E_VQMC - E0)
                if delta_E > 0:
                    v_score = N_spins * Varianza / (delta_E**2)
                else:
                    v_score = 0.0 # Ajuste perfecto
                    
                # Guardar fila de resultados
                resultados.append({
                    'Alpha': alpha,
                    'J': J,
                    'Muestreo': muestreo,
                    'E_0': E0,
                    'E_VQMC': round(E_VQMC, 4),
                    'Var(E)': round(Varianza, 6),
                    'E_rel': E_rel,
                    'E_rel (x10^-4)': round(E_rel * 10000, 2),
                    'V-Score': round(v_score, 4)
                })

    # Convertir a DataFrame para visualizarlo bonito y poder exportarlo
    df = pd.DataFrame(resultados)
    
    # Ordenar por J, luego por Alpha y Muestreo para que coincida con tu tabla
    df.sort_values(by=['J', 'Alpha', 'Muestreo'], inplace=True)
    
    return df

# Ejecutar y mostrar
if __name__ == "__main__":
    # Usamos la ruta absoluta exacta donde están los logs
    ruta_logs = "/content/TFG_ARViT/plots" 
    
    df_final = analizar_logs(ruta_logs)
    
    # Añadimos un pequeño control de seguridad por si acaso
    if df_final.empty:
        print(f"⚠️ No se ha encontrado ningún archivo .log válido en: {ruta_logs}")
        print("Asegúrate de que los archivos se copiaron correctamente a esa carpeta.")
    else:
        print(df_final.to_string(index=False))