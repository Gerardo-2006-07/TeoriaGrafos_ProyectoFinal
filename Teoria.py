import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def pedir_entero(mensaje, ejemplo):
    """Solicita un entero no negativo con validación robusta."""
    while True:
        entrada = input(mensaje).strip()
        if entrada == "":
            print(f" Error: Debes ingresar un número entero (ejemplo: {ejemplo}).")
            continue
        try:
            valor = int(entrada)
        except ValueError:
            print(f" Error: Ingresaste un valor no numérico. Intenta de nuevo con un entero (ejemplo: {ejemplo}).")
            continue
        if valor < 0:
            print(f" Error: El número no puede ser negativo. Intenta de nuevo (ejemplo: {ejemplo}).")
            continue
        return valor


def pedir_tipo_grafica():
    """Solicita el tipo de gráfica: dirigida o no dirigida."""
    while True:
        entrada = input("3. Tipo de gráfica (1 para dirigida, 2 para no dirigida): ").strip()
        if entrada == "1":
            return True
        if entrada == "2":
            return False
        print(" Error: Opción no válida. Ingresa solo '1' o '2'.")


def pedir_aristas(num_vertices, num_lineas):
    """Solicita los pares de vértices que definen cada línea/arista."""
    aristas = []
    if num_lineas == 0:
        return aristas

    print(f"\nPaso 3: Ingresa {num_lineas} pares de vértices. Ejemplo: 1 2")
    for i in range(num_lineas):
        while True:
            entrada = input(f"   Línea {i + 1}: ").strip()
            partes = entrada.split()
            if len(partes) != 2:
                print(" Error: Debes ingresar exactamente dos números separados por un espacio.")
                continue
            try:
                u = int(partes[0])
                v = int(partes[1])
            except ValueError:
                print(" Error: Uno de los valores no es un número entero válido.")
                continue
            if not (1 <= u <= num_vertices) or not (1 <= v <= num_vertices):
                print(f" Error: Los vértices deben estar entre 1 y {num_vertices}.")
                continue
            aristas.append((u - 1, v - 1))
            break
    return aristas


def construir_matrices(num_vertices, aristas, es_dirigida):
    """Construye las matrices de incidencia y adyacencia según las especificaciones.
    
    Matriz de Incidencia Dirigida: 1 para salida, -1 para llegada, '±1' para bucles.
    Matriz de Incidencia No Dirigida: Solo 1 y 0.
    Matriz de Adyacencia: Solo 1 y 0 (sin sumar líneas paralelas).
    """
    # Para dirigidas, usar dtype object para soportar '±1' en bucles
    if es_dirigida:
        incidencia = np.zeros((num_vertices, len(aristas)), dtype=object)
    else:
        incidencia = np.zeros((num_vertices, len(aristas)), dtype=int)
    
    adyacencia = np.zeros((num_vertices, num_vertices), dtype=int)

    for indice, (u, v) in enumerate(aristas):
        if es_dirigida:
            if u == v:
                # Bucle en gráfica dirigida: símbolo '±1'
                incidencia[u, indice] = '±1'
                adyacencia[u, u] = 1
            else:
                # Arista dirigida: 1 en salida, -1 en llegada
                incidencia[u, indice] = 1
                incidencia[v, indice] = -1
                adyacencia[u, v] = 1
        else:
            if u == v:
                # Bucle en gráfica no dirigida: un solo 1 en la incidencia
                incidencia[u, indice] = 1
                adyacencia[u, u] = 1
            else:
                # Arista no dirigida: 1s en ambos extremos
                incidencia[u, indice] = 1
                incidencia[v, indice] = 1
                adyacencia[u, v] = 1
                adyacencia[v, u] = 1
    return incidencia, adyacencia


def calcular_accesibilidad(adyacencia, es_dirigida):
    """Calcula la matriz de accesibilidad: A + A^2 + ... + A^k.
    
    Para gráficas dirigidas: itera hasta n-1
    Para gráficas no dirigidas: itera hasta (n²+n)/2
    Comienza directamente desde A^1 sin la matriz identidad.
    """
    n = adyacencia.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=int)
    
    # Inicializar directamente desde A^1 (no desde identidad)
    accesibilidad = np.copy(adyacencia)
    potencia = np.copy(adyacencia)
    
    # Determinar límite de iteración según tipo de gráfica
    if es_dirigida:
        """max_iter = n - 1"""
        max_iter = n
    else:
        max_iter = (n**2 + n) // 2
    
    # Acumular potencias sucesivas A^2, A^3, ... hasta A^max_iter
    for _ in range(1, max_iter):
        potencia = potencia.dot(adyacencia)
        accesibilidad += potencia
    
    return (accesibilidad > 0).astype(int)


def formatear_accesibilidad_para_impresion(accesibilidad):
    """Convierte la matriz de accesibilidad numérica a formato texto para impresión.
    
    Reemplaza 1 por '+' y mantiene '0' para mayor claridad visual.
    """
    if accesibilidad.size == 0:
        return accesibilidad
    accesibilidad_formateada = np.empty_like(accesibilidad, dtype=object)
    for i in range(accesibilidad.shape[0]):
        for j in range(accesibilidad.shape[1]):
            accesibilidad_formateada[i, j] = '+' if accesibilidad[i, j] == 1 else '0'
    return accesibilidad_formateada


def formatear_incidencia_dirigida_para_impresion(incidencia):
    """Convierte la matriz de incidencia dirigida a formato legible para impresión.
    
    Ya contiene valores numéricos y simbólicos (1, -1, '±1', 0).
    """
    if incidencia.size == 0:
        return incidencia
    incidencia_formateada = np.empty_like(incidencia, dtype=object)
    for i in range(incidencia.shape[0]):
        for j in range(incidencia.shape[1]):
            valor = incidencia[i, j]
            # Convertir 0 a string para consistencia
            if valor == 0:
                incidencia_formateada[i, j] = 0
            else:
                incidencia_formateada[i, j] = valor
    return incidencia_formateada


def grados_vectores(aristas, num_vertices, es_dirigida):
    """Calcula grados, in-degrees y out-degrees iterando sobre las aristas.
    
    No usa suma de matriz para evitar errores con líneas paralelas.
    """
    if es_dirigida:
        # Inicializar arreglos para grados de entrada y salida
        ingreso = np.zeros(num_vertices, dtype=int)
        salida = np.zeros(num_vertices, dtype=int)
        
        # Iterar sobre aristas
        for u, v in aristas:
            if u == v:
                # Bucle: cuenta tanto como entrada como salida
                salida[u] += 1
                ingreso[u] += 1
            else:
                # Arista dirigida: u es salida, v es llegada
                salida[u] += 1
                ingreso[v] += 1
        
        total = ingreso + salida
        return total, ingreso, salida
    else:
        # Inicializar arreglo de grados
        grado = np.zeros(num_vertices, dtype=int)
        
        # Iterar sobre aristas
        for u, v in aristas:
            if u == v:
                # Bucle: cuenta como 1 (no como 2)
                grado[u] += 1
            else:
                # Arista no dirigida: suma para ambos extremos
                grado[u] += 1
                grado[v] += 1
        
        return grado, None, None


def detectar_paralelas(incidencia, aristas):
    """Detecta líneas paralelas comparando columnas idénticas de la matriz de incidencia."""
    columnas = {}
    grupos = []
    for j in range(incidencia.shape[1]):
        clave = tuple(incidencia[:, j].tolist())
        columnas.setdefault(clave, []).append(j)
    for indices in columnas.values():
        if len(indices) > 1:
            grupos.append(indices)
    resultado = []
    for grupo in grupos:
        descripcion = [f"Línea {i+1} ({aristas[i][0] + 1}-{aristas[i][1] + 1})" for i in grupo]
        resultado.append(descripcion)
    return resultado


def detectar_lineas_serie(aristas, grados, es_dirigida):
    """Identifica aristas relacionadas con vértices de grado 2."""
    resultado = []
    for vertice, grado in enumerate(grados):
        if grado == 2:
            lineas = []
            for indice, (u, v) in enumerate(aristas):
                if u == vertice or v == vertice:
                    lineas.append(f"Línea {indice + 1} ({u + 1}-{v + 1})")
            if lineas:
                resultado.append((vertice + 1, lineas))
    return resultado


def es_conectada_por_accesibilidad(accesibilidad):
    """Determina conectividad si todos los valores de accesibilidad son 1."""
    if accesibilidad.size == 0:
        return False
    return np.all(accesibilidad == 1)


def es_completa_grafica(adyacencia, es_dirigida):
    """Determina si la gráfica es completa según la matriz de adyacencia."""
    n = adyacencia.shape[0]
    if n == 0:
        return False
    if es_dirigida:
        matriz_sin_diagonal = adyacencia.copy()
        np.fill_diagonal(matriz_sin_diagonal, 0)
        return np.all(matriz_sin_diagonal >= 1)
    else:
        if not np.array_equal(adyacencia, adyacencia.T):
            return False
        for i in range(n):
            for j in range(i + 1, n):
                if adyacencia[i, j] < 1:
                    return False
        return True


def es_arbol_grafica(num_vertices, num_lineas, es_dirigida, conectado):
    return not es_dirigida and conectado and num_lineas == max(0, num_vertices - 1)


def condiciones_eulerianas(num_vertices, aristas, es_dirigida, conectado):
    """Calcula si la gráfica tiene camino o circuito euleriano.
    
    Calcula grados iterando sobre aristas para evitar errores con líneas paralelas.
    """
    if num_vertices == 0:
        return False, False
    if es_dirigida:
        ingreso = np.zeros(num_vertices, dtype=int)
        salida = np.zeros(num_vertices, dtype=int)
        
        for u, v in aristas:
            if u == v:
                salida[u] += 1
                ingreso[u] += 1
            else:
                salida[u] += 1
                ingreso[v] += 1
        
        balanceado = np.array_equal(salida, ingreso)
        delta = salida - ingreso
        tiene_uno_mas = np.count_nonzero(delta == 1)
        tiene_menos_uno = np.count_nonzero(delta == -1)
        euleriano = conectado and balanceado and np.any(salida + ingreso > 0)
        unicursal = conectado and tiene_uno_mas == 1 and tiene_menos_uno == 1 and np.count_nonzero(delta == 0) == num_vertices - 2
        return euleriano, unicursal
    else:
        grado = np.zeros(num_vertices, dtype=int)
        
        for u, v in aristas:
            if u == v:
                grado[u] += 1
            else:
                grado[u] += 1
                grado[v] += 1
        
        impares = np.count_nonzero(grado % 2 == 1)
        euleriano = conectado and impares == 0 and np.any(grado > 0)
        unicursal = conectado and impares in (0, 2) and np.any(grado > 0)
        return euleriano, unicursal


"""FUNCIÓN PARA DIBUJAR LA GRÁFICA USANDO NETWORKX Y MATPLOTLIB"""
def dibujar_grafica(num_vertices, aristas, es_dirigida):
    """Dibuja la gráfica utilizando NetworkX y matplotlib."""
    if es_dirigida:
        G = nx.MultiDiGraph()
    else:
        G = nx.MultiGraph()

    G.add_nodes_from(range(1, num_vertices + 1))
    for u, v in aristas:
        G.add_edge(u + 1, v + 1)

    pos = nx.circular_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        arrows=es_dirigida,
        connectionstyle="arc3,rad=0.1",
        node_size=700,
        font_size=10,
    )
    plt.savefig("grafica_generada.png")
    plt.show()


"""""
def dibujar_grafica(num_vertices, aristas, es_dirigida):
    Dibuja la gráfica y etiqueta cada línea con su número de identificación.
    if es_dirigida:
        G = nx.MultiDiGraph()
    else:
        G = nx.MultiGraph()

    G.add_nodes_from(range(1, num_vertices + 1))
    
    # Agregar las aristas con un atributo 'id' para poder etiquetarlas
    for i, (u, v) in enumerate(aristas):
        G.add_edge(u + 1, v + 1, label=str(i + 1))

    pos = nx.circular_layout(G)
    
    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=700)
    nx.draw_networkx_labels(G, pos)
    
    # Dibujar aristas con una pequeña curvatura para que no se peguen al centro
    nx.draw_networkx_edges(
        G, pos, 
        edge_color="gray", 
        arrows=es_dirigida, 
        connectionstyle="arc3,rad=0.1",
        arrowstyle='->', 
        arrowsize=20
    )
    
    # EXTRAER Y DIBUJAR ETIQUETAS DE ARISTAS
    # Obtenemos los labels que definimos al añadir las aristas
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    
    # Nota: Si hay múltiples aristas entre los mismos nodos, 
    # esto mostrará las etiquetas encimadas. Para arreglarlo, 
    # usaremos una función que las distribuya un poco:
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Gráfica con Etiquetas de Líneas")
    plt.savefig("grafica_generada.png")
    plt.show()
"""""






def main():
    print("   ANALIZADOR DE GRÁFICAS  ")
    print("Integrantes: Gerardo Fajardo De la O, González Eslava Néstor Fabian, Jimenez Pulido Rodolfo Natanael")
    print("=========================================")
    print("Paso 1: Configuraremos el tamaño de tu gráfica.\n")

    num_vertices = pedir_entero("1. Número de Vértices: ", "5")
    num_lineas = pedir_entero("2. Número de Líneas (aristas): ", "4")
    es_dirigida = pedir_tipo_grafica()
    aristas = pedir_aristas(num_vertices, num_lineas)

    incidencia, adyacencia = construir_matrices(num_vertices, aristas, es_dirigida)
    accesibilidad = calcular_accesibilidad(adyacencia, es_dirigida)
    grados, ingreso, salida = grados_vectores(aristas, num_vertices, es_dirigida)
    conectado = es_conectada_por_accesibilidad(accesibilidad)
    paralelas = detectar_paralelas(incidencia, aristas)
    lineas_serie = detectar_lineas_serie(aristas, grados, es_dirigida)
    loops = sorted(list(set([u + 1 for u, v in aristas if u == v])))
    es_simple = len(paralelas) == 0 and len(loops) == 0
    es_regular = False
    if num_vertices > 0:
        if es_dirigida:
            es_regular = np.all(ingreso == ingreso[0]) and np.all(salida == salida[0])
        else:
            es_regular = np.all(grados == grados[0])
    es_completa = es_completa_grafica(adyacencia, es_dirigida)
    es_arbol = es_arbol_grafica(num_vertices, num_lineas, es_dirigida, conectado)
    euleriano, unicursal = condiciones_eulerianas(num_vertices, aristas, es_dirigida, conectado)
    simetrica = False
    balanceada = False
    if es_dirigida:
        simetrica = np.array_equal(adyacencia, adyacencia.T)
        balanceada = np.array_equal(ingreso, salida)

    print("\n" + "=" * 40)
    print("RESULTADOS")
    print("=" * 40)

    print("\n--- MATRICES ---")
    print("1. Matriz de Incidencia (Vértices x Líneas):")
    if incidencia.size:
        if es_dirigida:
            print(formatear_incidencia_dirigida_para_impresion(incidencia))
        else:
            print(incidencia)
    else:
        print(np.array([], dtype=int).reshape(num_vertices, 0))
    print("\n2. Matriz de Adyacencia (Vértices x Vértices):")
    print(adyacencia)
    print("\n3. Matriz de Accesibilidad:")
    print(formatear_accesibilidad_para_impresion(accesibilidad))

    print("\n--- INFORMACIÓN DE VÉRTICES ---")
    if num_vertices == 0:
        print("No hay vértices para analizar.")
    else:
        for vertice in range(num_vertices):
            etiqueta = vertice + 1
            if es_dirigida:
                print(f"   Vértice {etiqueta}: Entrada={ingreso[vertice]}, Salida={salida[vertice]}, Total={grados[vertice]}")
            else:
                print(f"   Vértice {etiqueta}: Grado={grados[vertice]}")

        aislados = [i + 1 for i, valor in enumerate(grados) if valor == 0]
        print(f"Aislados: {aislados if aislados else 'Ninguno'}")

        if not es_dirigida:
            colgantes = [i + 1 for i, valor in enumerate(grados) if valor == 1]
            print(f"Colgantes: {colgantes if colgantes else 'Ninguno'}")
        else:
            iniciales = [i + 1 for i in range(num_vertices) if ingreso[i] == 0 and salida[i] > 0]
            finales = [i + 1 for i in range(num_vertices) if salida[i] == 0 and ingreso[i] > 0]
            print(f"Iniciales: {iniciales if iniciales else 'Ninguno'}")
            print(f"Finales: {finales if finales else 'Ninguno'}")



    print("\n--- INFORMACIÓN DE LÍNEAS ---")
    if num_lineas == 0:
        print("No hay líneas para analizar.")
    else:
        for i, (u, v) in enumerate(aristas):
            if es_dirigida:
                print(f"  - Línea {i+1}: Sale del vértice {u+1} y entra al vértice {v+1}")
            else:
                print(f"  - Línea {i+1}: Conecta el vértice {u+1} con el vértice {v+1}")
        print()

        if paralelas:
            print("Líneas paralelas:")
            for grupo in paralelas:
                print("   - " + ", ".join(grupo))
        else:
            print("Líneas paralelas: Ninguna")

        print(f"Bucles: {loops if loops else 'Ninguno'}")

        if lineas_serie:
            print("Líneas en serie (vértices de grado 2):")
            for vertice, lista in lineas_serie:
                print(f"   Vértice {vertice}: {', '.join(lista)}")
        else:
            print("Líneas en serie: Ninguna")

    print("\n--- CLASIFICACIÓN DE LA GRAFICA ---")
    print(f"Simple o General: {'Simple' if es_simple else 'General'}")
    print(f"Nula: {'Sí' if num_lineas == 0 else 'No'}")
    print(f"Conectada: {'Sí' if conectado else 'No'}")
    print(f"Regular: {'Sí' if es_regular else 'No'}")
    print(f"Completa: {'Sí' if es_completa else 'No'}")
    print(f"Árbol: {'Sí' if es_arbol else 'No'}")
    if es_dirigida:
        print(f"Simétrica: {'Sí' if simetrica else 'No'}")
        print(f"Balanceada: {'Sí' if balanceada else 'No'}")
    print(f"Euleriana: {'Sí' if euleriano else 'No'}")
    print(f"Unicursal: {'Sí' if unicursal else 'No'}")

    dibujar_grafica(num_vertices, aristas, es_dirigida)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nOcurrió un error: {e}")

    input("\nPresiona Enter para salir...")