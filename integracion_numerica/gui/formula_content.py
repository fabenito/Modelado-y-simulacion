"""
Contenido de texto para las fórmulas de integración numérica.
Separado del código GUI para mejor mantenibilidad.
"""

FORMULA_TEXT_CONTENT = """
═══════════════════════════════════════════════════════════════════════════════
                    FÓRMULAS DE INTEGRACIÓN NUMÉRICA (NEWTON-COTES)
═══════════════════════════════════════════════════════════════════════════════

Donde: h = (b-a)/n,  n = número de subdivisiones,  ξ ∈ [a,b]


───────────────────────────────────────────────────────────────────────────────
1. REGLA DEL RECTÁNGULO/PUNTO MEDIO (Grado 0)
───────────────────────────────────────────────────────────────────────────────
   
   Fórmula:  I ≈ h∑f((xi + xi+1)/2)  (evaluación en punto medio)
   
   Error:    E = (b-a)³f''(ξ)/(24n²)


───────────────────────────────────────────────────────────────────────────────
2. REGLA TRAPEZOIDAL (Grado 1)
───────────────────────────────────────────────────────────────────────────────
   
   Fórmula:  I ≈ (h/2)[f(a) + f(b)]
   
   Error:    E = -(b-a)³f''(ξ)/(12n²)


───────────────────────────────────────────────────────────────────────────────
3. REGLA DE SIMPSON 1/3 (Grado 2)
───────────────────────────────────────────────────────────────────────────────
   
   Fórmula:  I ≈ (h/3)[f(a) + 4f((a+b)/2) + f(b)]
   
   Error:    E = -(b-a)⁵f⁽⁴⁾(ξ)/(180n⁴)


───────────────────────────────────────────────────────────────────────────────
4. REGLA DE SIMPSON 3/8 (Grado 3)
───────────────────────────────────────────────────────────────────────────────
   
   Fórmula:  I ≈ (3h/8)[f(x₀) + 3f(x₁) + 3f(x₂) + f(x₃)]
   
   Error:    E = -3(b-a)⁵f⁽⁴⁾(ξ)/(80n⁴)


───────────────────────────────────────────────────────────────────────────────
5. REGLA DE BOOLE (Grado 4)
───────────────────────────────────────────────────────────────────────────────
   
   Fórmula:      I ≈ (2h/45)[7f(x₀) + 32f(x₁) + 12f(x₂) + 32f(x₃) + 7f(x₄)]
   
   Coeficientes: {7, 32, 12, 32, 14, 32, 12, 32, 7} (patrón repetitivo)
   
   Error:        E = -8(b-a)⁷f⁽⁶⁾(ξ)/(945n⁶)


───────────────────────────────────────────────────────────────────────────────
6. MÉTODO ADAPTATIVO (Simpson Recursivo)
───────────────────────────────────────────────────────────────────────────────
   
   Estimación de error:  E_est = |S_h - S_2h|/15
   
   Criterio:            Si E_est < ε → aceptar; sino → dividir intervalo
   

═══════════════════════════════════════════════════════════════════════════════
💡 PRINCIPIO GENERAL:
   A mayor grado del polinomio interpolante → Mayor precisión
   Pero también → Mayor costo computacional y sensibilidad numérica
═══════════════════════════════════════════════════════════════════════════════

NOTAS IMPORTANTES:
• El método del Rectángulo es el más simple pero menos preciso
• Simpson 1/3 requiere n par, Simpson 3/8 requiere n múltiplo de 3
• Boole requiere n múltiplo de 4
• El método Adaptativo ajusta automáticamente la precisión
• Para funciones suaves, métodos de mayor grado son más eficientes
• Para funciones irregulares, el método adaptativo es recomendable
"""

# Información adicional de cada método para tooltips o ayuda contextual
METHOD_DESCRIPTIONS = {
    'rectangulo': {
        'name': 'Regla del Rectángulo/Punto Medio',
        'degree': 0,
        'formula': 'I ≈ h∑f((xi + xi+1)/2)',
        'error': 'E = (b-a)³f\'\'(ξ)/(24n²)',
        'description': 'Método más simple, evalúa función en punto medio de cada subintervalo.',
        'requirements': 'Ninguno especial',
        'best_for': 'Funciones suaves, cálculos rápidos'
    },
    'trapezoidal': {
        'name': 'Regla Trapezoidal', 
        'degree': 1,
        'formula': 'I ≈ (h/2)[f(a) + 2∑f(xi) + f(b)]',
        'error': 'E = -(b-a)³f\'\'(ξ)/(12n²)',
        'description': 'Aproxima la función con líneas rectas entre puntos.',
        'requirements': 'Ninguno especial',
        'best_for': 'Funciones lineales por tramos, uso general'
    },
    'simpson_13': {
        'name': 'Regla de Simpson 1/3',
        'degree': 2, 
        'formula': 'I ≈ (h/3)[f(a) + 4f(x₁) + 2f(x₂) + ... + f(b)]',
        'error': 'E = -(b-a)⁵f⁽⁴⁾(ξ)/(180n⁴)',
        'description': 'Usa parábolas para aproximar la función.',
        'requirements': 'n debe ser par',
        'best_for': 'Funciones cuadráticas, alta precisión'
    },
    'simpson_38': {
        'name': 'Regla de Simpson 3/8',
        'degree': 3,
        'formula': 'I ≈ (3h/8)[f(x₀) + 3f(x₁) + 3f(x₂) + f(x₃)]',
        'error': 'E = -3(b-a)⁵f⁽⁴⁾(ξ)/(80n⁴)',
        'description': 'Usa polinomios cúbicos para mayor precisión.',
        'requirements': 'n debe ser múltiplo de 3',
        'best_for': 'Funciones cúbicas, mejor que Simpson 1/3 para algunos casos'
    },
    'boole': {
        'name': 'Regla de Boole',
        'degree': 4,
        'formula': 'I ≈ (2h/45)[7f(x₀) + 32f(x₁) + 12f(x₂) + 32f(x₃) + 7f(x₄)]',
        'error': 'E = -8(b-a)⁷f⁽⁶⁾(ξ)/(945n⁶)',
        'description': 'Máxima precisión con polinomios de grado 4.',
        'requirements': 'n debe ser múltiplo de 4',
        'best_for': 'Funciones muy suaves, máxima precisión'
    },
    'adaptativo': {
        'name': 'Método Adaptativo (Simpson)',
        'degree': 2,
        'formula': 'Variable según error local',
        'error': 'E_est = |S_h - S_{2h}|/15',
        'description': 'Ajusta automáticamente la precisión dividiendo intervalos.',
        'requirements': 'Solo tolerancia de error',
        'best_for': 'Funciones irregulares, precisión garantizada'
    }
}

# Headers para diferentes contextos
HEADERS = {
    'main': 'FÓRMULAS DE INTEGRACIÓN NUMÉRICA (NEWTON-COTES)',
    'comparison': 'COMPARACIÓN DE MÉTODOS DE INTEGRACIÓN',
    'theory': 'FUNDAMENTOS TEÓRICOS - NEWTON-COTES'
}

# Notas pedagógicas adicionales
PEDAGOGICAL_NOTES = {
    'newton_cotes_principle': """
    🎓 PRINCIPIO DE NEWTON-COTES:
    Los métodos de Newton-Cotes se basan en aproximar la función con polinomios
    interpolantes usando puntos equidistantes. A mayor grado del polinomio:
    • Mayor precisión teórica
    • Mayor costo computacional  
    • Mayor sensibilidad a errores de redondeo
    """,
    
    'error_analysis': """
    📊 ANÁLISIS DE ERROR:
    • Grado 0-1: Error proporcional a h³ (derivada segunda)
    • Grado 2-3: Error proporcional a h⁵ (derivada cuarta)  
    • Grado 4: Error proporcional a h⁷ (derivada sexta)
    • Adaptativo: Error controlado por tolerancia del usuario
    """,
    
    'practical_advice': """
    💡 CONSEJOS PRÁCTICOS:
    • Para funciones suaves: Usar métodos de alto grado
    • Para funciones irregulares: Usar método adaptativo
    • Para cálculos rápidos: Regla del rectángulo o trapezoidal
    • Para máxima precisión: Boole (si n es múltiplo de 4)
    • Para uso general: Simpson 1/3 (buen balance precisión/costo)
    """
}
