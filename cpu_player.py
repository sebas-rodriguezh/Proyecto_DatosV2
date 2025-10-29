# entities/cpu_player.py
import pygame
import random
import math
from collections import deque
import heapq  # Agregar import para cola de prioridad
from entities.player import Player
from entities.order_list import OrderList
from entities.order import Order
from core.game_time import GameTime

class CPUPlayer(Player):
    """Jugador controlado por IA con tres niveles de dificultad"""
    
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
        
        # Para nivel difícil
        self.graph_representation = None
        self.known_obstacles = set()
        
        # Color diferente para distinguir del jugador humano
        self.cpu_color = (255, 0, 0)  # Rojo para la CPU
    
    def update(self, dt, active_orders, game_map, game_time, weather_system):
        """Actualiza el comportamiento de la IA según la dificultad"""
        # Actualizar movimiento primero
        self.update_movement(dt, weather_system.get_stamina_consumption())
        
        self.decision_cooldown -= dt
        self.replan_cooldown -= dt
        
        # Solo tomar decisiones si no está en medio de un movimiento
        if not self.is_moving and self.decision_cooldown <= 0:
            if self.difficulty == "easy":
                self._update_easy(active_orders, game_map)
            elif self.difficulty == "medium":
                self._update_medium(active_orders, game_map, game_time, weather_system)
            elif self.difficulty == "hard":
                self._update_hard(active_orders, game_map, game_time, weather_system)
            
            self.decision_cooldown = random.uniform(0.5, 1.5)
        
        # Ejecutar movimiento si hay un camino
        self._follow_path(dt, game_map, weather_system)
        
        # Manejar interacciones automáticamente
        self._handle_auto_interactions(active_orders, self.completed_orders, game_time, game_map)
    
    def _update_easy(self, active_orders, game_map):
        """IA Fácil: Comportamiento aleatorio - MEJORADO"""
        print(f"CPU Easy - Tomando decisión en ({self.grid_x}, {self.grid_y})...")
        
        # Primero verificar si hay pedidos que pueda recoger inmediatamente
        for order in active_orders:
            if (not self.inventory.find_by_id(order.id) and self.can_pickup_order(order)):
                pickup_x, pickup_y = order.pickup
                distance = max(abs(self.grid_x - pickup_x), abs(self.grid_y - pickup_y))
                
                # Si está muy cerca, ir por ese pedido
                if distance <= 2:
                    self.current_target = order.pickup
                    print(f"CPU Easy - Pedido cercano: Recoger {order.id} (distancia: {distance})")
                    self._generate_random_path(self.current_target, game_map)
                    return
        
        if not self.current_target or random.random() < 0.3 or not self.path:
            # Elegir un objetivo aleatorio
            available_orders = list(active_orders)
            
            if available_orders and random.random() < 0.8:  # Mayor probabilidad de elegir pedidos
                # Elegir un pedido aleatorio
                order = random.choice(available_orders)
                if not self.inventory.find_by_id(order.id) and self.can_pickup_order(order):
                    self.current_target = order.pickup
                    print(f"CPU Easy - Objetivo: Recoger {order.id} en {order.pickup}")
                elif self.inventory.find_by_id(order.id):
                    self.current_target = order.dropoff
                    print(f"CPU Easy - Objetivo {order.id} en {order.dropoff}")
            else:
                # Moverse a una posición aleatoria
                attempts = 0
                while attempts < 10:
                    random_x = random.randint(1, game_map.width - 2)
                    random_y = random.randint(1, game_map.height - 2)
                    if not game_map.legend.get(game_map.tiles[random_y][random_x], {}).get("blocked", False):
                        self.current_target = (random_x, random_y)
                        print(f"CPU Easy - Objetivo: Moverse a posición aleatoria {self.current_target}")
                        break
                    attempts += 1
            
            if self.current_target:
                self._generate_random_path(self.current_target, game_map)


    def get_interactable_orders(self, orders, game_map, radius=1, game_time=None):
        """Obtiene órdenes interactuables para la CPU - similar al del jugador humano"""
        interactable = []
        
        if game_time is None:
            return interactable
        else:
            current_time = game_time.get_current_game_time()
        
        # Pedidos para RECOGER
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
        
        # Pedidos para ENTREGAR
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


    def _update_medium(self, active_orders, game_map, game_time, weather_system):
        """IA Medio: Evaluación greedy con horizonte limitado - MEJORADO"""
        print(f"CPU Medium - Evaluando opciones desde posición ({self.grid_x}, {self.grid_y})...")
        
        # Usar COLA DE PRIORIDAD (heap) para evaluar mejores opciones
        options = []
        
        current_time = game_time.get_current_game_time()
        
        # Evaluar secuencias de 2-3 acciones (horizonte limitado)
        for order in active_orders:
            if not self.inventory.find_by_id(order.id) and self.can_pickup_order(order):
                score = self._evaluate_action_sequence(['pickup', 'deliver'], order, current_time, game_map, weather_system)
                heapq.heappush(options, (-score, 'pickup', order))  # Max-heap usando negativo
        
        # Evaluar entregas en inventario
        for order in self.inventory:
            score = self._evaluate_single_action('deliver', order, current_time, game_map, weather_system)
            heapq.heappush(options, (-score, 'deliver', order))
        
        # Elegir la mejor opción
        if options:
            best_score, best_action, best_order = heapq.heappop(options)
            best_score = -best_score  # Convertir de vuelta a positivo
            
            if best_action == 'pickup':
                target_pos = self._get_nearest_accessible_position(best_order.pickup, game_map)
            else:
                target_pos = self._get_nearest_accessible_position(best_order.dropoff, game_map)
            
            if target_pos:
                self.current_target = target_pos
                self._generate_direct_path(self.current_target, game_map)
                print(f"🎯 CPU Medium - Elegido: {best_action} {best_order.id} (score: {best_score:.1f})")
                return
        
        # Movimiento exploratorio si no hay buenas opciones
        print("CPU Medium - Sin objetivos buenos, movimiento exploratorio")
        self._exploratory_move(game_map)

    def _evaluate_action_sequence(self, actions, order, current_time, game_map, weather_system):
        """Evalúa una secuencia de acciones con horizonte limitado - ÁRBOL DE DECISIÓN SIMPLE"""
        total_score = 0
        current_pos = (self.grid_x, self.grid_y)
        
        for i, action in enumerate(actions):
            if action == 'pickup':
                target = order.pickup
                score_func = self._evaluate_order_pickup
            else:  # 'deliver'
                target = order.dropoff
                score_func = self._evaluate_order_delivery
            
            accessible_pos = self._get_nearest_accessible_position(target, game_map)
            if not accessible_pos:
                return -float('inf')
            
            distance = self._manhattan_distance(current_pos[0], current_pos[1], 
                                            accessible_pos[0], accessible_pos[1])
            
            score = score_func(order, current_time, game_map, weather_system, distance)
            
            # Aplicar descuento exponencial para acciones futuras
            discount_factor = 0.7 ** i
            total_score += score * discount_factor
            
            # Actualizar posición estimada para la siguiente acción
            current_pos = accessible_pos
        
        return total_score

    def _evaluate_single_action(self, action, order, current_time, game_map, weather_system):
        """Evalúa una acción única"""
        if action == 'pickup':
            return self._evaluate_order_pickup(order, current_time, game_map, weather_system)
        else:  # 'deliver'
            return self._evaluate_order_delivery(order, current_time, game_map, weather_system)

    def _update_hard(self, active_orders, game_map, game_time, weather_system):
        """IA Difícil: Optimización basada en grafos"""
        if self.replan_cooldown <= 0:
            print("CPU Hard - Replanificando ruta óptima...")
            self._plan_optimal_route(active_orders, game_map, game_time, weather_system)
            self.replan_cooldown = 3.0
        
        # Si no hay camino, generar uno
        if not self.path and self.current_target:
            self._generate_astar_path(self.current_target, game_map)
    
    def _evaluate_order_pickup(self, order, current_time, game_map, weather_system, distance=None):
        """Función de evaluación para recoger pedidos - CORREGIDA"""
        # Calcular distancia a la posición accesible
        accessible_position = self._get_nearest_accessible_position(order.pickup, game_map)
        if not accessible_position:
            return -float('inf')
        
        if distance is None:
            distance = self._manhattan_distance(self.grid_x, self.grid_y, 
                                            accessible_position[0], accessible_position[1])
        
        time_remaining = order.get_time_remaining(current_time)
        
        if time_remaining <= 0:
            return -float('inf')
        
        # Calcular distancia de entrega también
        dropoff_accessible = self._get_nearest_accessible_position(order.dropoff, game_map)
        if dropoff_accessible:
            delivery_distance = self._manhattan_distance(accessible_position[0], accessible_position[1],
                                                dropoff_accessible[0], dropoff_accessible[1])
        else:
            delivery_distance = 10  # Distancia por defecto si no se encuentra posición accesible
        
        total_dist = distance + delivery_distance
        
        # Factor de tiempo (penalización por poco tiempo)
        time_factor = min(1.0, time_remaining / 300)
        
        # Puntuación base con factores
        base_score = order.payout
        priority_bonus = order.priority * 50
        distance_penalty = total_dist * 2
        time_bonus = time_factor * 100
        
        score = base_score + priority_bonus + time_bonus - distance_penalty
        
        return score

    def _evaluate_order_delivery(self, order, current_time, game_map, weather_system, distance=None):
        """Función de evaluación para entregar pedidos - CORREGIDA"""
        # Calcular distancia a la posición accesible
        accessible_position = self._get_nearest_accessible_position(order.dropoff, game_map)
        if not accessible_position:
            return -float('inf')
        
        if distance is None:
            distance = self._manhattan_distance(self.grid_x, self.grid_y, 
                                            accessible_position[0], accessible_position[1])
        
        time_remaining = order.get_time_remaining(current_time)
        
        if time_remaining <= 0:
            return -float('inf')
        
        # Factores
        base_score = order.payout * 1.2  # Bonus por entrega
        priority_bonus = order.priority * 75
        time_bonus = min(200, (time_remaining / 60) * 50)
        distance_penalty = distance * 1.5
        
        score = base_score + priority_bonus + time_bonus - distance_penalty
        
        return score
    
    def _plan_optimal_route(self, active_orders, game_map, game_time, weather_system):
        """Planificación de ruta óptima (nivel difícil) - MEJORADO"""
        if not active_orders and self.inventory.is_empty():
            self._exploratory_move(game_map)
            return
        
        current_time = game_time.get_current_game_time()
        best_option = None
        best_score = -float('inf')
        
        # Evaluar pedidos para recoger
        for order in active_orders:
            if not self.inventory.find_by_id(order.id) and self.can_pickup_order(order):
                score = self._evaluate_order_pickup(order, current_time, game_map, weather_system)
                
                if score > best_score:
                    best_score = score
                    best_option = ('pickup', order)
        
        # Evaluar entregas en inventario
        for order in self.inventory:
            score = self._evaluate_order_delivery(order, current_time, game_map, weather_system)
            
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
                self._generate_astar_path(self.current_target, game_map)
                print(f"🎯 CPU Hard - Elegido: {action_type} {order.id} (score: {best_score:.1f})")
                return
        
        # Fallback: movimiento exploratorio
        print("CPU Hard - Sin objetivos óptimos, movimiento exploratorio")
        self._exploratory_move(game_map)
    
    def _evaluate_sequence(self, pickup_orders, current_time, game_map, delivery_order=None):
        """Evalúa una secuencia de acciones (nivel difícil)"""
        total_score = 0
        current_pos = (self.grid_x, self.grid_y)
        
        # Calcular coste de recogidas
        for order in pickup_orders:
            pickup_dist = self._a_star_distance(current_pos, order.pickup, game_map)
            current_pos = order.pickup
            time_cost = pickup_dist * 10  # Convertir distancia a tiempo aproximado
            
            if order.get_time_remaining(current_time) - time_cost <= 0:
                return -float('inf')
            
            total_score += order.payout - time_cost
        
        # Calcular coste de entrega
        if delivery_order:
            delivery_dist = self._a_star_distance(current_pos, delivery_order.dropoff, game_map)
            time_cost = delivery_dist * 10
            
            if delivery_order.get_time_remaining(current_time) - time_cost <= 0:
                return -float('inf')
            
            total_score += delivery_order.payout * 1.2 - time_cost
        
        return total_score
    
    def _generate_random_path(self, target, game_map):
        """Ahora usa el método directo para niveles fáciles también"""
        self._generate_direct_path(target, game_map)
        
    def _generate_greedy_path(self, target, game_map):
        """Ahora usa el método directo"""
        self._generate_direct_path(target, game_map)
    
    def _generate_astar_path(self, target, game_map):
        """Genera camino usando algoritmo A* (nivel difícil) - CORREGIDO"""
        self.path.clear()
        
        start = (self.grid_x, self.grid_y)
        goal = target
        
        # Implementación completa de A*
        open_set = []
        heapq.heappush(open_set, (0, start))  # (f_score, position)
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: self._manhattan_distance(start[0], start[1], goal[0], goal[1])}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                self._reconstruct_path(came_from, current)
                print(f"✓ Camino A* generado: {len(self.path)} pasos")
                return
            
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Verificar validez del vecino
                if not (0 <= neighbor[0] < game_map.width and 0 <= neighbor[1] < game_map.height):
                    continue
                
                if game_map.legend.get(game_map.tiles[neighbor[1]][neighbor[0]], {}).get("blocked", False):
                    continue
                
                # Coste del movimiento (podría variar según el terreno)
                tentative_g_score = g_score[current] + 1
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._manhattan_distance(neighbor[0], neighbor[1], goal[0], goal[1])
                    
                    # Añadir a open_set si no está
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # Si A* falla, usar camino simple como fallback
        print("❌ A* falló, usando camino simple")
        self._generate_simple_path(target, game_map)
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruye el camino desde el diccionario came_from"""
        total_path = [current]
        
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        
        # Invertir y quitar la posición actual
        total_path.reverse()
        if total_path and total_path[0] == (self.grid_x, self.grid_y):
            total_path = total_path[1:]
        
        self.path = deque(total_path)
    
    def _follow_path(self, dt, game_map, weather_system):
        """Sigue el camino generado - VERSIÓN MEJORADA"""
        if self.difficulty == "medium":
            self._follow_path_improved(dt, game_map, weather_system)
        else:  # hard (ya usa A* que es robusto)
            # Mantener el código original para hard
            if not self.path:
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
        """Movimiento exploratorio INTELIGENTE para evitar obstáculos"""
        # ✅ Buscar dirección "menos obstructiva"
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        valid_directions = []
        
        for dx, dy in directions:
            new_x, new_y = self.grid_x + dx, self.grid_y + dy
            if (0 <= new_x < game_map.width and 0 <= new_y < game_map.height and
                not game_map.legend.get(game_map.tiles[new_y][new_x], {}).get("blocked", False)):
                valid_directions.append((dx, dy))
        
        if valid_directions:
            # ✅ Elegir dirección que nos aleje de obstáculos
            dx, dy = random.choice(valid_directions)
            self.current_target = (self.grid_x + dx * 3, self.grid_y + dy * 3)  # Meta a 3 pasos
            # Asegurar que esté dentro de los límites
            self.current_target = (
                max(1, min(self.current_target[0], game_map.width - 2)),
                max(1, min(self.current_target[1], game_map.height - 2))
            )
            self._generate_direct_path(self.current_target, game_map)
            print(f"🔀 CPU Medium - Movimiento evasivo hacia {self.current_target}")
        else:
            print("❌ CPU Medium - Completamente bloqueado, esperando...")
    
    def _follow_path_improved(self, dt, game_map, weather_system):
        """Seguimiento de camino MEJORADO para dificultad medium"""
        if not self.path:
            return
        
        # ✅ VERIFICACIÓN COMPLETA del camino restante
        invalid_index = -1
        for i, pos in enumerate(self.path):
            if (pos[0] < 0 or pos[0] >= game_map.width or 
                pos[1] < 0 or pos[1] >= game_map.height or
                game_map.legend.get(game_map.tiles[pos[1]][pos[0]], {}).get("blocked", False)):
                invalid_index = i
                break
        
        # ❌ Si hay obstáculos en el camino, replanificar COMPLETAMENTE
        if invalid_index != -1:
            print(f"⚠️ Camino bloqueado en paso {invalid_index}, replanificando ruta completa...")
            self.path.clear()
            
            # ✅ ESTRATEGIA MEJORADA: Buscar posición accesible alternativa
            accessible_target = self._get_nearest_accessible_position(self.current_target, game_map)
            if accessible_target and accessible_target != (self.grid_x, self.grid_y):
                self.current_target = accessible_target
                self._generate_direct_path(self.current_target, game_map)
            else:
                # ✅ FALLBACK: Movimiento exploratorio inteligente
                self._smart_exploratory_move(game_map)
            return
        
        # Movimiento normal (código existente)...
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
                print(f"CPU Medium se movió a ({self.grid_x}, {self.grid_y})")
    def _exploratory_move(self, game_map):
        """Movimiento exploratorio cuando no hay objetivos claros - MEJORADO"""
        # Para nivel hard, buscar zonas con alta densidad de pedidos
        if self.difficulty == "hard" and hasattr(self, 'active_orders') and len(self.active_orders) > 0:
            # Buscar el centro de masa de los pedidos activos
            pickup_positions = [order.pickup for order in self.active_orders]
            if pickup_positions:
                center_x = sum(pos[0] for pos in pickup_positions) // len(pickup_positions)
                center_y = sum(pos[1] for pos in pickup_positions) // len(pickup_positions)
                
                # Buscar posición válida cerca del centro
                for radius in range(1, 6):
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            test_pos = (center_x + dx, center_y + dy)
                            if (0 <= test_pos[0] < game_map.width and 
                                0 <= test_pos[1] < game_map.height and
                                not game_map.legend.get(game_map.tiles[test_pos[1]][test_pos[0]], {}).get("blocked", False)):
                                
                                self.current_target = test_pos
                                self._generate_astar_path(self.current_target, game_map)
                                print(f"CPU Hard - Exploración inteligente hacia {self.current_target}")
                                return
        
        # Fallback: posición aleatoria (comportamiento original)
        attempts = 0
        while attempts < 10:
            random_x = random.randint(1, game_map.width - 2)
            random_y = random.randint(1, game_map.height - 2)
            if not game_map.legend.get(game_map.tiles[random_y][random_x], {}).get("blocked", False):
                self.current_target = (random_x, random_y)
                print(f"CPU - Movimiento exploratorio a {self.current_target}")
                if self.difficulty == "hard":
                    self._generate_astar_path(self.current_target, game_map)
                else:
                    self._generate_direct_path(self.current_target, game_map)
                break
            attempts += 1

    def _generate_simple_path(self, target, game_map):
        """Genera un camino simple cuando BFS falla - MÉTODO NUEVO"""
        print("⚠️ Usando camino simple como fallback")
        self.path.clear()
        
        start_x, start_y = self.grid_x, self.grid_y
        target_x, target_y = target
        
        current_x, current_y = start_x, start_y
        
        # Algoritmo simple de movimiento directo
        max_steps = 50
        steps = 0
        
        while (current_x, current_y) != (target_x, target_y) and steps < max_steps:
            # Decidir dirección preferida
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
            
            # Priorizar movimiento horizontal primero, luego vertical
            if dx != 0:
                new_x, new_y = current_x + dx, current_y
            else:
                new_x, new_y = current_x, current_y + dy
            
            # Verificar si la nueva posición es válida
            if (0 <= new_x < game_map.width and 0 <= new_y < game_map.height and
                not game_map.legend.get(game_map.tiles[new_y][new_x], {}).get("blocked", False)):
                
                self.path.append((new_x, new_y))
                current_x, current_y = new_x, new_y
            else:
                # Intentar dirección alternativa
                if dx != 0 and dy != 0:
                    # Intentar movimiento vertical en lugar de horizontal
                    alt_x, alt_y = current_x, current_y + dy
                    if (0 <= alt_x < game_map.width and 0 <= alt_y < game_map.height and
                        not game_map.legend.get(game_map.tiles[alt_y][alt_x], {}).get("blocked", False)):
                        
                        self.path.append((alt_x, alt_y))
                        current_x, current_y = alt_x, alt_y
                    else:
                        # Intentar movimiento horizontal en lugar de vertical
                        alt_x, alt_y = current_x + dx, current_y
                        if (0 <= alt_x < game_map.width and 0 <= alt_y < game_map.height and
                            not game_map.legend.get(game_map.tiles[alt_y][alt_x], {}).get("blocked", False)):
                            
                            self.path.append((alt_x, alt_y))
                            current_x, current_y = alt_x, alt_y
                        else:
                            break  # No hay movimientos posibles
                else:
                    break  # No hay movimientos posibles
            
            steps += 1
        
        print(f"✅ Camino simple generado: {len(self.path)} pasos")

    def _handle_auto_interactions(self, active_orders, completed_orders, game_time, game_map):
        """Maneja interacciones automáticas - CORREGIDO PARA EDIFICIOS"""
        current_time = game_time.get_current_game_time()
        
        # Verificar recogida de pedidos
        for order in list(active_orders):
            if (not order.is_expired and not order.is_completed and 
                not order.is_in_inventory and not self.inventory.find_by_id(order.id)):
                
                # Encontrar posición accesible para este pedido
                accessible_position = self._get_nearest_accessible_position(order.pickup, game_map)
                
                if accessible_position:
                    distance_to_accessible = max(abs(self.grid_x - accessible_position[0]), 
                                            abs(self.grid_y - accessible_position[1]))
                    
                    print(f"🔍 Verificando pedido {order.id}")
                    print(f"   Edificio pickup: {order.pickup}")
                    print(f"   Posición accesible: {accessible_position}")
                    print(f"   Distancia a posición accesible: {distance_to_accessible}")
                    
                    # Permitir recoger desde la posición accesible o adyacente
                    if distance_to_accessible <= 1:
                        if self.can_pickup_order(order):
                            if self.add_to_inventory(order):
                                order.mark_as_picked_up()
                                order.mark_as_accepted(current_time)
                                active_orders.remove_by_id(order.id)
                                print(f"✅ CPU RECOGIÓ pedido: {order.id}")
                                print(f"   Desde posición accesible: {accessible_position}")
                                
                                # Actualizar objetivo a la entrega
                                dropoff_accessible = self._get_nearest_accessible_position(order.dropoff, game_map)
                                if dropoff_accessible:
                                    self.current_target = dropoff_accessible
                                    self.path.clear()
                                    self._generate_direct_path(self.current_target, game_map)
                                return
                        else:
                            print(f"❌ No puede recoger {order.id} - Sin capacidad")
        
        # Verificar entrega de pedidos
        for order in list(self.inventory):
            # Encontrar posición accesible para la entrega
            accessible_position = self._get_nearest_accessible_position(order.dropoff, game_map)
            
            if accessible_position:
                distance_to_accessible = max(abs(self.grid_x - accessible_position[0]), 
                                        abs(self.grid_y - accessible_position[1]))
                
                print(f"🔍 Verificando entrega {order.id}")
                print(f"   Edificio dropoff: {order.dropoff}")
                print(f"   Posición accesible: {accessible_position}")
                print(f"   Distancia a posición accesible: {distance_to_accessible}")
                
                # Permitir entregar desde la posición accesible o adyacente
                if distance_to_accessible <= 1:
                    if self.remove_from_inventory(order.id):
                        # Calcular ganancias
                        earnings = order.payout
                        reputation_change = order.calculate_reputation_change(current_time)
                        
                        self.total_earnings += earnings
                        self.reputation = min(100, max(0, self.reputation + reputation_change))
                        
                        order.mark_as_completed()
                        completed_orders.enqueue(order)
                        self.orders_completed += 1
                        
                        print(f"✅ CPU ENTREGÓ pedido: {order.id} (+${earnings})")
                        print(f"   Desde posición accesible: {accessible_position}")
                        
                        # Limpiar objetivos
                        self.path.clear()
                        self.current_target = None
                        return
    
    def _manhattan_distance(self, x1, y1, x2, y2):
        """Calcula distancia Manhattan entre dos puntos"""
        return abs(x1 - x2) + abs(y1 - y2)
    
    def _a_star_distance(self, start, goal, game_map):
        """Calcula distancia REAL usando A* entre dos puntos - CORREGIDO"""
        # Implementación real que usa el algoritmo A*
        if start == goal:
            return 0
            
        open_set = []
        heapq.heappush(open_set, (0, start))
        g_score = {start: 0}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                return g_score[current]
                
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
        
        return float('inf')  # No se encontró camino
    def draw(self, screen, camera_x=0, camera_y=0):
        """Dibuja al jugador CPU con color diferente"""
        # Llamar al método draw del padre pero con color diferente
        super().draw(screen, camera_x, camera_y)
        
        # Dibujar un indicador adicional para la CPU
        screen_x = (self.grid_x - camera_x) * self.tile_size
        screen_y = (self.grid_y - camera_y) * self.tile_size
        
        # Dibujar un círculo rojo alrededor de la CPU
        pygame.draw.circle(screen, self.cpu_color, 
                         (int(screen_x + self.tile_size // 2), 
                          int(screen_y + self.tile_size // 2)), 
                         self.tile_size // 2 + 2, 2)
        
    def _find_adjacent_position(self, building_position, game_map):
        """Encuentra una posición válida adyacente a un edificio"""
        building_x, building_y = building_position
        
        # Buscar en las 4 direcciones cardinales
        adjacent_positions = [
            (building_x + 1, building_y),    # Derecha
            (building_x - 1, building_y),    # Izquierda  
            (building_x, building_y + 1),    # Abajo
            (building_x, building_y - 1),    # Arriba,
        ]
        
        # Buscar posiciones válidas (no bloqueadas y dentro del mapa)
        valid_positions = []
        for pos_x, pos_y in adjacent_positions:
            if (0 <= pos_x < game_map.width and 0 <= pos_y < game_map.height and
                not game_map.legend.get(game_map.tiles[pos_y][pos_x], {}).get("blocked", False)):
                valid_positions.append((pos_x, pos_y))
        
        return valid_positions

    def _get_nearest_accessible_position(self, target_position, game_map):
        """Encuentra la posición accesible más cercana a un edificio - MEJORADO"""
        if not target_position:
            return None
            
        building_x, building_y = target_position
        
        # Primero verificar si la posición objetivo ya es accesible (no es un edificio)
        if (0 <= building_x < game_map.width and 0 <= building_y < game_map.height and
            not game_map.legend.get(game_map.tiles[building_y][building_x], {}).get("blocked", False)):
            return (building_x, building_y)
        
        # Buscar en las 4 direcciones cardinales
        adjacent_positions = [
            (building_x + 1, building_y),    # Derecha
            (building_x - 1, building_y),    # Izquierda  
            (building_x, building_y + 1),    # Abajo
            (building_x, building_y - 1),    # Arriba
        ]
        
        # Buscar posiciones válidas (no bloqueadas y dentro del mapa)
        valid_positions = []
        for pos_x, pos_y in adjacent_positions:
            if (0 <= pos_x < game_map.width and 0 <= pos_y < game_map.height and
                not game_map.legend.get(game_map.tiles[pos_y][pos_x], {}).get("blocked", False)):
                valid_positions.append((pos_x, pos_y))
        
        if valid_positions:
            # Encontrar la más cercana a la posición actual
            closest_pos = None
            min_distance = float('inf')
            
            for pos in valid_positions:
                distance = self._manhattan_distance(self.grid_x, self.grid_y, pos[0], pos[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_pos = pos
            
            return closest_pos
        
        # Si no hay posiciones adyacentes válidas, buscar en un radio mayor
        for radius in range(2, 6):  # Buscar en radios de 2 a 5 casillas
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                        
                    pos_x, pos_y = building_x + dx, building_y + dy
                    if (0 <= pos_x < game_map.width and 0 <= pos_y < game_map.height and
                        not game_map.legend.get(game_map.tiles[pos_y][pos_x], {}).get("blocked", False)):
                        
                        return (pos_x, pos_y)
        
        print(f"❌ No se encontró posición accesible para {target_position}")
        return None  # No se encontró posición accesible
    
    def _generate_direct_path(self, target, game_map):
        """Genera un camino válido usando BFS - ESTRUCTURA DE COLA"""
        self.path.clear()
        
        start = (self.grid_x, self.grid_y)
        goal = target
        
        # Usar BFS para encontrar camino válido
        queue = deque()  # COLA para BFS
        queue.append((start, []))  # (posición actual, camino hasta aquí)
        visited = set()
        visited.add(start)
        
        max_iterations = 200
        iterations = 0
        
        while queue and iterations < max_iterations:
            current_pos, current_path = queue.popleft()
            
            if current_pos == goal:
                self.path = deque(current_path)
                print(f"✅ Camino BFS generado: {len(self.path)} pasos")
                return
            
            # Explorar vecinos en 4 direcciones
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current_pos[0] + dx, current_pos[1] + dy)
                
                # Verificar validez del vecino
                if (neighbor not in visited and 
                    0 <= neighbor[0] < game_map.width and 
                    0 <= neighbor[1] < game_map.height and
                    not game_map.legend.get(game_map.tiles[neighbor[1]][neighbor[0]], {}).get("blocked", False)):
                    
                    visited.add(neighbor)
                    new_path = current_path + [neighbor]
                    queue.append((neighbor, new_path))
            
            iterations += 1
        
        # Fallback: camino simple si BFS falla
        self._generate_simple_path(target, game_map)