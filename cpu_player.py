# entities/cpu_player.py
import pygame
import random
import math
from collections import deque
import heapq
from entities.player import Player
from entities.order_list import OrderList
from entities.order import Order
from core.game_time import GameTime

class CPUPlayer(Player):
    """
    Jugador controlado por IA con tres niveles de dificultad
    Hereda de la clase Player y añade comportamientos automatizados para competir contra el jugador humano. Implementa tres estrategias de inteligencia artificial:

    - FÁCIL: Comportamiento semi-aleatorio con decisiones simples y búsqueda de camino simple.
    - MEDIO: Evaluación estratégica usando parámetros α, β, γ para optimizar ganancias
    - DIFÍCIL: Optimización avanzada con algoritmo A* y planificación de rutas óptimas
    """
    
    def __init__(self, x, y, tile_size, legend, difficulty="easy", scale_factor=1):
        super().__init__(x, y, tile_size, legend, scale_factor)
        self.difficulty = difficulty
        self.current_target = None
        self.path = deque()
        self.decision_cooldown = 0
        self.replan_cooldown = 0
        self.last_decision_time = 0
        
        # Estadísticas de la IA
        self.orders_completed = 0
        self.total_earnings = 0
        
        # Parámetros para nivel medio (α, β, γ)
        self.alpha = 1.0    # Peso del pago esperado
        self.beta = 0.5     # Peso del costo de distancia  
        self.gamma = 0.3    # Peso de penalización por clima
        
        # Para nivel difícil
        self.graph_representation = None
        self.known_obstacles = set()
        self.cpu_color = (255, 0, 0)
    
    def update(self, dt, active_orders, game_map, game_time, weather_system):
        """
        Actualiza el comportamiento de la IA según la dificultad
        La idea es que este método pueda recibir por parámetro, en base a esto, se actualiza la dificultad de la IA, según el caso seleccionado por el jugador real.
        Complejidad: O(1)
        """
        # Actualizar movimiento primero
        self.update_movement(dt, weather_system.get_stamina_consumption())
        
        self.decision_cooldown -= dt
        self.replan_cooldown -= dt
        
        # Acá toma la decisión de la IA, es decir se setea el estado en el cuál va a jugar. 
        if not self.is_moving and self.decision_cooldown <= 0:
            if self.difficulty == "easy":
                self._update_easy(active_orders, game_map)
            elif self.difficulty == "medium":
                self._update_medium(active_orders, game_map, game_time, weather_system)
            elif self.difficulty == "hard":
                self._update_hard(active_orders, game_map, game_time, weather_system)
            
            self.decision_cooldown = random.uniform(0.5, 1.5)
        
        # Ejecutar movimiento, si hay camino. 
        self._follow_path(dt, game_map, weather_system)
        
        # Manejar interacciones de manera automática.
        self._handle_auto_interactions(active_orders, self.completed_orders, game_time, game_map)
    
    def _update_easy(self, active_orders, game_map):
        """
        IA Fácil: Comportamiento aleatorio usando Random Walk
        La idea es que pueda tener un comportamiento simple, pseudo aleatorio, pero que cuando logre "llegarle" a un pedido, pueda entregarlo.
        La idea es que el agente no sea inútil o incapaz de entregar pedidos.
        Complejidad: O(n)
        """
        # Valida si en el campo que spawmeo, hay pedidos cercanos para recoger.
        for order in active_orders:
            if (not self.inventory.find_by_id(order.id) and self.can_pickup_order(order)): #Verifica que NO lo tenga en su inventario de IA.
                pickup_x, pickup_y = order.pickup
                distance = max(abs(self.grid_x - pickup_x), abs(self.grid_y - pickup_y))
                
                # Si está muy cerca, ir por ese pedido
                if distance <= 2:
                    self.current_target = order.pickup
                    self._generate_direct_path(self.current_target, game_map) #Acá es donde se genera la ruta, para que pueda dejar el pedido. 
                    return
        
        if not self.current_target or random.random() < 0.3 or not self.path:
            available_orders = list(active_orders)
            
            if available_orders and random.random() < 0.8:  
                # Elegir un pedido aleatorio
                order = random.choice(available_orders)
                if not self.inventory.find_by_id(order.id) and self.can_pickup_order(order):
                    self.current_target = order.pickup
                elif self.inventory.find_by_id(order.id):
                    self.current_target = order.dropoff
            else:
                # Acá se mueve aleatoriamente. 
                attempts = 0
                while attempts < 10:
                    random_x = random.randint(1, game_map.width - 2)
                    random_y = random.randint(1, game_map.height - 2)
                    if not game_map.legend.get(game_map.tiles[random_y][random_x], {}).get("blocked", False):
                        self.current_target = (random_x, random_y)
                        break
                    attempts += 1
            
            if self.current_target:
                self._generate_direct_path(self.current_target, game_map)

    def _update_medium(self, active_orders, game_map, game_time, weather_system):
        """
        IA Medio: Evaluación greedy con función heurística usando α, β, γ
        Usamos el greedy por conveniencia, además utilizamos parametros α, β, γ para ponderar distintos aspectos de la evaluación.
        Complejidad: O(n log n)
        """
        # Usamos cola de prioridad (heap) 
        options = [] #Almacena mejores opciones. 
        
        current_time = game_time.get_current_game_time()
        
        # Evaluar secuencias de 2-3 acciones.
        for order in active_orders:
            if not self.inventory.find_by_id(order.id) and self.can_pickup_order(order):
                score = self._evaluate_action_sequence(['pickup', 'deliver'], order, current_time, game_map, weather_system)
                heapq.heappush(options, (-score, 'pickup', order))  # Max-heap usando negativo
        
        for order in self.inventory:
            score = self._evaluate_single_action('deliver', order, current_time, game_map, weather_system) #Acá es donde se evaluan las acciones (Prioriza).
            heapq.heappush(options, (-score, 'deliver', order))
        
        # Elegir la mejor opción
        if options:
            best_score, best_action, best_order = heapq.heappop(options)
            best_score = -best_score  # Convertir de vuelta a positivo (Para que seleccione la mejor y no la menor).
            
            if best_action == 'pickup':
                target_pos = self._get_nearest_accessible_position(best_order.pickup, game_map) #Este método es para la posición accesible y evitar obstáculos (builds).
            else:
                target_pos = self._get_nearest_accessible_position(best_order.dropoff, game_map)
            
            if target_pos:
                self.current_target = target_pos
                self._generate_direct_path(self.current_target, game_map) #Crea el camino. 
                return
            
        self._exploratory_move(game_map)  # Movimiento exploratorio si no hay mejores opciones


    def _evaluate_action_sequence(self, actions, order, current_time, game_map, weather_system):
        """
        Evalúa una secuencia de acciones con horizonte limitado - CON α, β, γ
        Este método solo usa el nivel medio. Para cumplir con lo del score y la evaluación de múltiples acciones.
        Complejidad: O(1)
        """
        total_score = 0
        current_pos = (self.grid_x, self.grid_y)
        
        for i, action in enumerate(actions):
            if action == 'pickup':
                target = order.pickup
                score_func = self._evaluate_order_pickup #Para evaluar si vale la pena recoger el pedido. Es formula. 
            else:  # Entrega
                target = order.dropoff
                score_func = self._evaluate_order_delivery #Para distancia entre entrega. Es formula. 
            
            accessible_pos = self._get_nearest_accessible_position(target, game_map)
            if not accessible_pos:
                return -float('inf') #Devuelve menos infinito si no es accesible.
            
            distance = self._manhattan_distance(current_pos[0], current_pos[1], 
                                            accessible_pos[0], accessible_pos[1]) #Calcula distancia entre puntos.
            
            score = score_func(order, current_time, game_map, weather_system, distance)
            
            # Factores para importancia.
            discount_factor = 0.7 ** i
            total_score += score * discount_factor
            
            # Empieza donde termino. 
            current_pos = accessible_pos
        
        return total_score

    def _evaluate_single_action(self, action, order, current_time, game_map, weather_system):
        """
        Evalúa una acción única
        Es como un puente para evaluar a donde irse. 
        Complejidad: O(1)
        """
        if action == 'pickup':
            return self._evaluate_order_pickup(order, current_time, game_map, weather_system)
        else:  
            return self._evaluate_order_delivery(order, current_time, game_map, weather_system)

    def _evaluate_order_pickup(self, order, current_time, game_map, weather_system, distance=None):
        """
        Función de evaluación para recoger pedidos - CON α, β, γ
        En resumen lo que hace es evaluar si vale la pena o no recoger un pedido, en base a la fórmula con α, β, γ, pero en base a un score.
        """
        # Calcular distancia a la posición accesible
        accessible_position = self._get_nearest_accessible_position(order.pickup, game_map)
        if not accessible_position:
            return -float('inf')
        
        if distance is None:
            distance = self._manhattan_distance(self.grid_x, self.grid_y, 
                                            accessible_position[0], accessible_position[1]) #Distancia
        
        time_remaining = order.get_time_remaining(current_time) #Usamos el tiempo para también que sea parametro en el score. 
        
        if time_remaining <= 0:
            return -float('inf')
        
        # Calcular distancia de entrega también
        dropoff_accessible = self._get_nearest_accessible_position(order.dropoff, game_map)
        if dropoff_accessible:
            delivery_distance = self._manhattan_distance(accessible_position[0], accessible_position[1],
                                                dropoff_accessible[0], dropoff_accessible[1])
        else:
            delivery_distance = 10
        
        total_dist = distance + delivery_distance
        
        # Factor de tiempo.
        time_factor = min(1.0, time_remaining / 300)
        
        # Acá se aplica la formula del enunciado α, β, γ
        expected_payout = order.payout
        distance_cost = total_dist * 2
        weather_penalty = (1.0 - weather_system.get_speed_multiplier()) * 50
        
        base_score = (self.alpha * expected_payout - 
                     self.beta * distance_cost - 
                     self.gamma * weather_penalty)
        
        # Adicionales para hacerlo más realista.
        priority_bonus = order.priority * 50
        time_bonus = time_factor * 100
        
        score = base_score + priority_bonus + time_bonus
        
        return score 

    def _evaluate_order_delivery(self, order, current_time, game_map, weather_system, distance=None):
        """
        Función de evaluación para entregar pedidos - CON α, β, γ
        La misma idea que el método anterior, solo que este evalúa que tan viable es hacer la entrega. 
        Complejidad: O(1)
        """

        accessible_position = self._get_nearest_accessible_position(order.dropoff, game_map)
        if not accessible_position:
            return -float('inf')
        
        if distance is None:
            distance = self._manhattan_distance(self.grid_x, self.grid_y, 
                                            accessible_position[0], accessible_position[1])
        
        time_remaining = order.get_time_remaining(current_time)
        
        if time_remaining <= 0:
            return -float('inf')
        
        expected_payout = order.payout * 1.2 
        distance_cost = distance * 1.5
        weather_penalty = (1.0 - weather_system.get_speed_multiplier()) * 50
        
        base_score = (self.alpha * expected_payout - 
                     self.beta * distance_cost - 
                     self.gamma * weather_penalty)
        
        priority_bonus = order.priority * 75
        time_bonus = min(200, (time_remaining / 60) * 50)
        
        score = base_score + priority_bonus + time_bonus
        
        return score

    def _update_hard(self, active_orders, game_map, game_time, weather_system):
        """
        IA Difícil: Optimización basada en grafos
        Hacemos uso del A* (Astar), para que el agente sea más inteligente en sus movimientos y pueda planificar mejor sus rutas.
        La idea es que la IA pueda tomar mejores decisiones que los casos anteriores. 
        Complejidad: O(1) Porque es solo verificación y replanificación
        """

        # Replanificar ruta si es necesario
        if self.replan_cooldown <= 0:
            self._plan_optimal_route(active_orders, game_map, game_time, weather_system) #Rutas y exploración. 
            self.replan_cooldown = 3.0
        
        # Si no hay camino, generar uno
        if not self.path and self.current_target:
            self._generate_astar_path(self.current_target, game_map) #Usamos el camino A*
    

    def _plan_optimal_route(self, active_orders, game_map, game_time, weather_system):
        """
        Planificación de ruta óptima (nivel difícil)
        Utiliza el algoritmo A* para planificar la ruta óptima hacia el siguiente objetivo basado en una evaluación completa de pedidos activos y entregas en inventario.
        Complejidad: O(n × A*)
        """
        if not active_orders and self.inventory.is_empty():
            self._exploratory_move(game_map)
            return
        
        current_time = game_time.get_current_game_time()
        best_option = None
        best_score = -float('inf')
        
        # Evaluar pedidos para recoger
        for order in active_orders:
            if not self.inventory.find_by_id(order.id) and self.can_pickup_order(order):
                score = self._evaluate_with_graph_distance(order, current_time, game_map, weather_system, 'pickup') #Usa grafo para calculos de distancia. 
                
                if score > best_score:
                    best_score = score
                    best_option = ('pickup', order)
        
        # Evaluar entregas en inventario
        for order in self.inventory:
            score = self._evaluate_with_graph_distance(order, current_time, game_map, weather_system, 'deliver')
            
            if score > best_score:
                best_score = score
                best_option = ('deliver', order)
        
        if best_option:
            action_type, order = best_option
            if action_type == 'pickup':
                target_pos = self._get_nearest_accessible_position(order.pickup, game_map)
            else:
                target_pos = self._get_nearest_accessible_position(order.dropoff, game_map)
            
            if target_pos:
                self.current_target = target_pos
                self._generate_astar_path(self.current_target, game_map) #Genera camino A*
                return
        
        self._exploratory_move(game_map) #Se mueve "exporatoriamente" si no hay mejores opciones.

    def _evaluate_with_graph_distance(self, order, current_time, game_map, weather_system, action_type):
        """
        Evaluación usando distancias reales del grafo
        Traduce el mapa en un grafo para calcular distancias reales y mejorar la evaluación de acciones.
        Este mecanismo se apoya en el A* para calcular distancias reales.
        Complejidad: O(A*)
        """

        if action_type == 'pickup':
            target = order.pickup
            base_payout = order.payout
        else:
            target = order.dropoff
            base_payout = order.payout * 1.2
        
        accessible_position = self._get_nearest_accessible_position(target, game_map)
        if not accessible_position:
            return -float('inf')
        
        # Usa distancia real usando A*
        real_distance = self._a_star_distance((self.grid_x, self.grid_y), accessible_position, game_map)
        if real_distance == float('inf'):
            return -float('inf')
        
        # Tiempo estimado necesario para llegar, afectado por clima
        time_remaining = order.get_time_remaining(current_time)
        if time_remaining <= 0:
            return -float('inf')
        
        # Considerar clima 
        weather_multiplier = weather_system.get_speed_multiplier()
        time_cost = real_distance * 10 / weather_multiplier
        
        if time_remaining - time_cost <= 0:
            return -float('inf')
        
        score = (base_payout + 
                order.priority * 100 + 
                min(200, (time_remaining / 60) * 50) - 
                real_distance * 3)
        
        return score

    def get_interactable_orders(self, orders, game_map, radius=1, game_time=None):
        """
        Obtiene órdenes interactuables para la CPU - similar al del jugador humano
        Detecta qué pedidos pueden ser recogidos o entregados en ese momento basado en la posición actual.
        Complejidad: O(n log n)
        """
        interactable = []
        
        if game_time is None:
            return interactable
        else:
            current_time = game_time.get_current_game_time()
        
        # Pedidos para recoger
        for order in orders:
            if (not order.is_expired and 
                not order.is_completed and 
                not order.is_in_inventory and
                not self.inventory.find_by_id(order.id)):
                
                if order.check_expiration(current_time):
                    continue
                
                can_pickup = self.is_near_location(order.pickup, include_exact=True, radius=radius)
                
                if can_pickup:
                    distance = max(abs(self.grid_x - order.pickup[0]), 
                                abs(self.grid_y - order.pickup[1]))
                    interactable.append({
                        'order': order,
                        'action': 'pickup',
                        'location': order.pickup,
                        'is_exact': self.is_at_location(order.pickup),
                        'distance': distance,
                        'is_building': self.is_building_location(order.pickup, game_map)
                    })
        
        # Pedidos para entregar
        for order in self.inventory:
            if not order.is_completed:
                if order.check_expiration(current_time):
                    continue
                
                can_dropoff = self.is_near_location(order.dropoff, include_exact=True, radius=radius)
                
                if can_dropoff:
                    distance = max(abs(self.grid_x - order.dropoff[0]), 
                                abs(self.grid_y - order.dropoff[1]))
                    interactable.append({
                        'order': order,
                        'action': 'dropoff',
                        'location': order.dropoff,
                        'is_exact': self.is_at_location(order.dropoff),
                        'distance': distance,
                        'is_building': self.is_building_location(order.dropoff, game_map)
                    })
        
        interactable.sort(key=lambda x: (not x['is_exact'], x['distance']))
        return interactable

    def _generate_astar_path(self, target, game_map):
        """
        Genera camino usando algoritmo A* (nivel difícil)
        sta función implementa A* para generar y guardar en self.path el camino desde la posición actual hasta target en el grid del juego.
        Complejidad: O(b^d) donde b = branching factor, d = profundidad del nodo objetivo
        """
        self.path.clear() #Limpia el camino previo.
        
        start = (self.grid_x, self.grid_y) #Origen / destino.
        goal = target
        
        # Implementación completa de A*

        open_set = [] #Tupla de prioridad
        heapq.heappush(open_set, (0, start)) #El heap asegura que siempre se extraiga el nodo con menor f.score
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: self._manhattan_distance(start[0], start[1], goal[0], goal[1])}
        
        #Saca el menor f.score
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                self._reconstruct_path(came_from, current) #Si llega al objetivo, reconstruye el camino.
                return
            
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Verifica validez del vecino
                if not (0 <= neighbor[0] < game_map.width and 0 <= neighbor[1] < game_map.height):
                    continue
                
                if game_map.legend.get(game_map.tiles[neighbor[1]][neighbor[0]], {}).get("blocked", False):
                    continue
                
                # Coste del movimiento
                tentative_g_score = g_score[current] + 1
                
                #Cuando upd el camino, como priorizar exploración y donde buscar a continuación. 
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._manhattan_distance(neighbor[0], neighbor[1], goal[0], goal[1])
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # Si A* falla, usar camino simple.
        self._generate_simple_path(target, game_map)

    def _generate_direct_path(self, target, game_map):
        """
        Genera un camino válido usando BFS
        Encuentra camino más corto en pasos, pero explora mucho más el área que el de A*.
        Este es utilizado por la dificultad medium para generar caminos directos.
        Las rutas que tome el medium dependen de este path generado. 
        Compleji9dad: O(V + E) donde V = nodos, E = aristas
        """
        self.path.clear()
        
        start = (self.grid_x, self.grid_y) #Inicia
        goal = target #Llega
        
        queue = deque() #Cola porque como es BFS.
        queue.append((start, [])) #Agrega tupla con (posición_inicial, camino_vacío)
        visited = set() #Evita revisitar nodos.
        visited.add(start) #Marcarla como visitada.
        
        max_iterations = 200 #Máximo de pasos, aunque casi nunca llega ahí. 
        iterations = 0
        

        #Mientras haya nodos por revsar, sacamos el más antiguo, evaluamos con posición actual y por último lista de pasos para llegar ahí. 
        while queue and iterations < max_iterations:
            current_pos, current_path = queue.popleft()
            
            if current_pos == goal:
                self.path = deque(current_path)
                return
            
            # Explorar vecinos en 4 direcciones
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current_pos[0] + dx, current_pos[1] + dy)
                
                if (neighbor not in visited and 
                    0 <= neighbor[0] < game_map.width and 
                    0 <= neighbor[1] < game_map.height and
                    not game_map.legend.get(game_map.tiles[neighbor[1]][neighbor[0]], {}).get("blocked", False)):
                    
                    visited.add(neighbor)
                    new_path = current_path + [neighbor]
                    queue.append((neighbor, new_path))
            
            iterations += 1
        
        # Si fallamos BFS, usar camino simple.
        self._generate_simple_path(target, game_map)

    def _generate_simple_path(self, target, game_map):
            """
            Genera un camino simple moviéndose directamente hacia el objetivo.

            Complejidad: O(1)
            """
            self.path.clear()
            
            start_x, start_y = self.grid_x, self.grid_y
            target_x, target_y = target
            
            current_x, current_y = start_x, start_y
            max_steps = 50
            steps = 0
            
            while (current_x, current_y) != (target_x, target_y) and steps < max_steps:
                dx = 0
                dy = 0
                
                if current_x < target_x:
                    dx = 1
                elif current_x > target_x:
                    dx = -1
                
                if current_y < target_y:
                    dy = 1
                elif current_y > target_y:
                    dy = -1
                
                if dx != 0:
                    new_x, new_y = current_x + dx, current_y
                else:
                    new_x, new_y = current_x, current_y + dy
                
                if (0 <= new_x < game_map.width and 0 <= new_y < game_map.height and
                    not game_map.legend.get(game_map.tiles[new_y][new_x], {}).get("blocked", False)):
                    
                    self.path.append((new_x, new_y))
                    current_x, current_y = new_x, new_y
                else:
                    if dx != 0 and dy != 0:
                        alt_x, alt_y = current_x, current_y + dy
                        if (0 <= alt_x < game_map.width and 0 <= alt_y < game_map.height and
                            not game_map.legend.get(game_map.tiles[alt_y][alt_x], {}).get("blocked", False)):
                            
                            self.path.append((alt_x, alt_y))
                            current_x, current_y = alt_x, alt_y
                        else:
                            alt_x, alt_y = current_x + dx, current_y
                            if (0 <= alt_x < game_map.width and 0 <= alt_y < game_map.height and
                                not game_map.legend.get(game_map.tiles[alt_y][alt_x], {}).get("blocked", False)):
                                
                                self.path.append((alt_x, alt_y))
                                current_x, current_y = alt_x, alt_y
                            else:
                                break
                    else:
                        break
                
                steps += 1

    def _follow_path(self, dt, game_map, weather_system):
        """
        Sigue el camino generado hacia el objetivo.
        
        Complejidad: O(1) para fácil, O(n) para medio donde n = largo del camino
        """
        if self.difficulty == "medium":
            self._follow_path_improved(dt, game_map, weather_system)
        else:
            if not self.path:
                return
            
            next_pos = self.path[0]
            dx = next_pos[0] - self.grid_x
            dy = next_pos[1] - self.grid_y
            
            # Convierte diferencia a dirección (-1, 0, 1)
            dx = 1 if dx > 0 else -1 if dx < 0 else 0
            dy = 1 if dy > 0 else -1 if dy < 0 else 0
            
            if dx != 0 or dy != 0:
                weather_multiplier = weather_system.get_speed_multiplier()
                tile_char = game_map.tiles[self.grid_y][self.grid_x]
                surface_multiplier = game_map.legend.get(tile_char, {}).get("surface_weight", 1.0)
                
                if self.try_move(dx, dy, game_map.tiles, weather_multiplier, surface_multiplier):
                    self.path.popleft()

    def _follow_path_improved(self, dt, game_map, weather_system):
        """
        Versión mejorada para nivel medio que verifica si el camino sigue siendo válido.
        
        Complejidad: O(n) donde n = largo del camino
        """
        if not self.path:
            return
        
        # Verifica si hay obstáculos en el camino
        invalid_index = -1
        for i, pos in enumerate(self.path):
            if (pos[0] < 0 or pos[0] >= game_map.width or 
                pos[1] < 0 or pos[1] >= game_map.height or
                game_map.legend.get(game_map.tiles[pos[1]][pos[0]], {}).get("blocked", False)):
                invalid_index = i
                break
        
        if invalid_index != -1:
            self.path.clear()
            accessible_target = self._get_nearest_accessible_position(self.current_target, game_map)
            if accessible_target and accessible_target != (self.grid_x, self.grid_y):
                self.current_target = accessible_target
                self._generate_direct_path(self.current_target, game_map)
            else:
                self._smart_exploratory_move(game_map)
            return
        
        next_pos = self.path[0]
        dx = next_pos[0] - self.grid_x
        dy = next_pos[1] - self.grid_y
        
        dx = 1 if dx > 0 else -1 if dx < 0 else 0
        dy = 1 if dy > 0 else -1 if dy < 0 else 0
        
        if dx != 0 or dy != 0:
            weather_multiplier = weather_system.get_speed_multiplier()
            tile_char = game_map.tiles[self.grid_y][self.grid_x]
            surface_multiplier = game_map.legend.get(tile_char, {}).get("surface_weight", 1.0)
            
            if self.try_move(dx, dy, game_map.tiles, weather_multiplier, surface_multiplier):
                self.path.popleft()

    def _smart_exploratory_move(self, game_map):
        """
        Movimiento inteligente para evitar obstáculos cuando el camino está bloqueado.
        
        Complejidad: O(1)
        """
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        valid_directions = []
        
        # Encuentra direcciones válidas
        for dx, dy in directions:
            new_x, new_y = self.grid_x + dx, self.grid_y + dy
            if (0 <= new_x < game_map.width and 0 <= new_y < game_map.height and
                not game_map.legend.get(game_map.tiles[new_y][new_x], {}).get("blocked", False)):
                valid_directions.append((dx, dy))
        
        if valid_directions:
            dx, dy = random.choice(valid_directions)
            # Establece objetivo 3 casillas en dirección elegida
            self.current_target = (self.grid_x + dx * 3, self.grid_y + dy * 3)
            self.current_target = (
                max(1, min(self.current_target[0], game_map.width - 2)),
                max(1, min(self.current_target[1], game_map.height - 2))
            )
            self._generate_direct_path(self.current_target, game_map)

    def _exploratory_move(self, game_map):
        """
        Movimiento de exploración cuando no hay objetivos claros.
        
        Complejidad: O(n) donde n = número de pedidos activos
        """
        # Para nivel difícil, busca centro de pedidos activos
        if self.difficulty == "hard" and hasattr(self, 'active_orders') and len(self.active_orders) > 0:
            pickup_positions = [order.pickup for order in self.active_orders]
            if pickup_positions:
                center_x = sum(pos[0] for pos in pickup_positions) // len(pickup_positions)
                center_y = sum(pos[1] for pos in pickup_positions) // len(pickup_positions)
                
                # Busca posición válida alrededor del centro
                for radius in range(1, 6):
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            test_pos = (center_x + dx, center_y + dy)
                            if (0 <= test_pos[0] < game_map.width and 
                                0 <= test_pos[1] < game_map.height and
                                not game_map.legend.get(game_map.tiles[test_pos[1]][test_pos[0]], {}).get("blocked", False)):
                                
                                self.current_target = test_pos
                                self._generate_astar_path(self.current_target, game_map)
                                return
        
        # Fallback: posición aleatoria
        attempts = 0
        while attempts < 10:
            random_x = random.randint(1, game_map.width - 2)
            random_y = random.randint(1, game_map.height - 2)
            if not game_map.legend.get(game_map.tiles[random_y][random_x], {}).get("blocked", False):
                self.current_target = (random_x, random_y)
                if self.difficulty == "hard":
                    self._generate_astar_path(self.current_target, game_map)
                else:
                    self._generate_direct_path(self.current_target, game_map)
                break
            attempts += 1

    def _handle_auto_interactions(self, active_orders, completed_orders, game_time, game_map):
        """
        Maneja las interacciones automáticas: recoger y entregar pedidos.
        
        Complejidad: O(m + n) donde m = pedidos activos, n = pedidos en inventario
        """
        current_time = game_time.get_current_game_time()
        
        # Verifica recogida de pedidos
        for order in list(active_orders):
            if (not order.is_expired and not order.is_completed and 
                not order.is_in_inventory and not self.inventory.find_by_id(order.id)):
                
                accessible_position = self._get_nearest_accessible_position(order.pickup, game_map)
                
                if accessible_position:
                    distance_to_accessible = max(abs(self.grid_x - accessible_position[0]), 
                                            abs(self.grid_y - accessible_position[1]))
                    
                    # Si está cerca del punto de recogida
                    if distance_to_accessible <= 1:
                        if self.can_pickup_order(order):
                            if self.add_to_inventory(order):
                                order.mark_as_picked_up()
                                order.mark_as_accepted(current_time)
                                active_orders.remove_by_id(order.id)
                                
                                # Establece entrega como nuevo objetivo
                                dropoff_accessible = self._get_nearest_accessible_position(order.dropoff, game_map)
                                if dropoff_accessible:
                                    self.current_target = dropoff_accessible
                                    self.path.clear()
                                    self._generate_direct_path(self.current_target, game_map)
                                return
        
        # Verifica entrega de pedidos
        for order in list(self.inventory):
            accessible_position = self._get_nearest_accessible_position(order.dropoff, game_map)
            
            if accessible_position:
                distance_to_accessible = max(abs(self.grid_x - accessible_position[0]), 
                                        abs(self.grid_y - accessible_position[1]))
                
                if distance_to_accessible <= 1:
                    if self.remove_from_inventory(order.id):
                        reputation_change = order.calculate_reputation_change(current_time)
                        multiplicador_earnings = order.calculate_payout_modifier(current_time, reputation_change)
                        earnings = order.payout * multiplicador_earnings
                        self.total_earnings += order.payout * multiplicador_earnings
                        self.reputation = min(100, max(0, self.reputation + reputation_change))
                        
                        order.mark_as_completed()
                        completed_orders.enqueue(order)
                        self.orders_completed += 1
                        
                        self.path.clear()
                        self.current_target = None
                        return

    def _a_star_distance(self, start, goal, game_map):
        """
        Calcula la distancia usando el algoritmo A* entre dos puntos.
        
        Complejidad: O(n log n) donde n = número de nodos visitados
        """
        if start == goal:
            return 0
            
        open_set = []
        heapq.heappush(open_set, (0, start))
        g_score = {start: 0}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                return g_score[current]
                
            # Revisa los 4 vecinos
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < game_map.width and 0 <= neighbor[1] < game_map.height):
                    continue
                    
                if game_map.legend.get(game_map.tiles[neighbor[1]][neighbor[0]], {}).get("blocked", False):
                    continue
                    
                tentative_g_score = g_score[current] + 1
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._manhattan_distance(neighbor[0], neighbor[1], goal[0], goal[1])
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return float('inf')

    def _get_nearest_accessible_position(self, target_position, game_map):
        """
        Encuentra la posición accesible más cercana a un edificio.
        
        Complejidad: O(1) en promedio, O(r²) en peor caso donde r = radio de búsqueda
        """
        if not target_position:
            return None
            
        building_x, building_y = target_position
        
        # Si ya es accesible
        if (0 <= building_x < game_map.width and 0 <= building_y < game_map.height and
            not game_map.legend.get(game_map.tiles[building_y][building_x], {}).get("blocked", False)):
            return (building_x, building_y)
        
        # Busca en posiciones adyacentes
        adjacent_positions = [
            (building_x + 1, building_y),
            (building_x - 1, building_y),  
            (building_x, building_y + 1),
            (building_x, building_y - 1),
        ]
        
        valid_positions = []
        for pos_x, pos_y in adjacent_positions:
            if (0 <= pos_x < game_map.width and 0 <= pos_y < game_map.height and
                not game_map.legend.get(game_map.tiles[pos_y][pos_x], {}).get("blocked", False)):
                valid_positions.append((pos_x, pos_y))
        
        if valid_positions:
            closest_pos = None
            min_distance = float('inf')
            
            for pos in valid_positions:
                distance = self._manhattan_distance(self.grid_x, self.grid_y, pos[0], pos[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_pos = pos
            
            return closest_pos
        
        # Busca en radio más grande si no encuentra adyacentes
        for radius in range(2, 6):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                        
                    pos_x, pos_y = building_x + dx, building_y + dy
                    if (0 <= pos_x < game_map.width and 0 <= pos_y < game_map.height and
                        not game_map.legend.get(game_map.tiles[pos_y][pos_x], {}).get("blocked", False)):
                        
                        return (pos_x, pos_y)
        
        return None
    
    def _manhattan_distance(self, x1, y1, x2, y2):
        """
        Calcula la distancia Manhattan entre dos puntos en una cuadrícula.

        Complejidad: O(1)
        """
        return abs(x1 - x2) + abs(y1 - y2)
    
    def _reconstruct_path(self, came_from, current):
        """
        Reconstruye el camino desde el objetivo hasta el inicio.

        Complejidad: O(n) donde n = largo del camino
        """
        total_path = [current]

        while current in came_from:
            current = came_from[current]
            total_path.append(current)

        total_path.reverse()
        if total_path and total_path[0] == (self.grid_x, self.grid_y):
            total_path = total_path[1:]

        self.path = deque(total_path)