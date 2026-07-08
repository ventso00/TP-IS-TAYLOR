
"""
TP : Approximation polynomiale par la méthode de Taylor (Ordre 3)
Fichier : taylor_tp.py
"""

import numpy as np

# --- Constantes du circuit ---
C = 0.22e-6   # F
R = 4700      # Ohms
L = 150e-3    # H
alpha = -0.06 * np.pi
omega = 2.0

# --- Fonctions pour E(t) et ses dérivées successives ---
def E(t):
    return -np.exp(alpha * t) * np.sin(omega * t)

def dE(t):
    return alpha * E(t) - omega * np.exp(alpha * t) * np.cos(omega * t)

def d2E(t):
    return (alpha**2 - omega**2) * E(t) - 2 * alpha * omega * np.exp(alpha * t) * np.cos(omega * t)

def d3E(t):
    return alpha * (alpha**2 - 3 * omega**2) * E(t) - omega * (3 * alpha**2 - omega**2) * np.exp(alpha * t) * np.cos(omega * t)

def d4E(t):
    return (alpha**4 - 6 * alpha**2 * omega**2 + omega**4) * E(t) - 4 * alpha * omega * (alpha**2 - omega**2) * np.exp(alpha * t) * np.cos(omega * t)

# --- Dérivées de i(t) via l'équation différentielle ---
def di_dt(t):
    return C * d2E(t) + (1.0 / R) * dE(t) + (1.0 / L) * E(t)

def d2i_dt2(t):
    return C * d3E(t) + (1.0 / R) * d2E(t) + (1.0 / L) * dE(t)

def d3i_dt3(t):
    return C * d4E(t) + (1.0 / R) * d3E(t) + (1.0 / L) * d2E(t)

# --- Solution Analytique Exacte pour le calcul de l'erreur ---
def i_exact(t):
    # Primitive de e^(alpha*t)*sin(omega*t)
    def prim_sin(u):
        return (np.exp(alpha * u) / (alpha**2 + omega**2)) * (alpha * np.sin(omega * u) - omega * np.cos(omega * u))
    
    # Intégrale de E(u) entre 0 et t
    int_E = -(prim_sin(t) - prim_sin(0))
    return C * dE(t) + (1.0 / R) * E(t) + (1.0 / L) * int_E - (C * dE(0) + (1.0 / R) * E(0))

# --- Résolution par Taylor Ordre 3 ---
def solve_taylor_3(t_max, step_size, num_steps):
    t_vals = np.zeros(num_steps + 1)
    i_vals = np.zeros(num_steps + 1)
    
    # Conditions initiales
    t_vals[0] = 0.0
    i_vals[0] = 0.0
    
    h = step_size
    for j in range(num_steps):
        t = t_vals[j]
        i = i_vals[j]
        
        # Algorithme de Taylor d'ordre 3
        i_next = i + h * di_dt(t) + ((h**2) / 2.0) * d2i_dt2(t) + ((h**3) / 6.0) * d3i_dt3(t)
        
        t_vals[j+1] = t + h
        i_vals[j+1] = i_next
        
    return t_vals, i_vals

# --- Question 1 ---
print("Exécution de la Question 1...")
t1, i1 = solve_taylor_3(t_max=10.0, step_size=0.1, num_steps=100)
with open("resultats_question1.txt", "w") as f:
    f.write(f"{'t':<12}{'i(t)':<25}\n")
    f.write("-" * 40 + "\n")
    for t, i in zip(t1, i1):
        f.write(f"{t:<12.2f}{i:<25.16e}\n")

# --- Question 2 ---
print("Exécution de la Question 2...")
t2, i2 = solve_taylor_3(t_max=2.0, step_size=0.01, num_steps=200)
with open("resultats_question2.txt", "w") as f:
    f.write(f"{'t':<12}{'i(t)':<25}{'erreur':<25}\n")
    f.write("-" * 65 + "\n")
    for t, i in zip(t2, i2):
        exact = i_exact(t)
        erreur_abs = abs(i - exact)
        f.write(f"{t:<12.2f}{i:<25.16e}{erreur_abs:<25.16e}\n")

print("Fichiers 'resultats_question1.txt' et 'resultats_question2.txt' générés avec succès.")
