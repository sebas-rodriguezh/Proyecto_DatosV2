# entities/cpu_player.py
import pygame
import random
import math
from collections import deque
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
                    print(f"CPU Easy - Objetivo: Entregar {order.id} en {order.dropoff}")
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
    
    def _update_medium(self, active_orders, game_map, game_time, weather_system):
        """IA Medio: Evaluación greedy con horizonte limitado - CON MANEJO DE ERRORES"""
        print(f"CPU Medium - Evaluando opciones desde posición ({self.grid_x}, {self.grid_y})...")
        
        best_score = -float('inf')
        best_action = None
        best_order = None
        best_accessible_position = None
        
        current_time = game_time.get_current_game_time()
        
        try:
            # Evaluar pedidos disponibles
            for order in active_orders:
                if not self.inventory.find_by_id(order.id) and self.can_pickup_order(order):
                    # Encontrar posición accesible para recoger
                    accessible_position = self._get_nearest_accessible_position(order.pickup, game_map)
                    
                    if accessible_position:
                        # Calcular distancia a la posición accesible, no al edificio
                        distance = self._manhattan_distance(self.grid_x, self.grid_y, 
                                                        accessible_position[0], accessible_position[1])
                        
                        # Evaluar con la posición accesible
                        score = self._evaluate_order_pickup(order, current_time, game_map, weather_system, distance)
                        print(f"  Pedido {order.id} - Pickup: {order.pickup} -> Accesible: {accessible_position} - Score: {score:.1f}")
                        
                        if score > best_score:
                            best_score = score
                            best_action = 'pickup'
                            best_order = order
                            best_accessible_position = accessible_position
                    else:
                        print(f"  Pedido {order.id} - Sin posición accesible para pickup {order.pickup}")
            
            # Evaluar entregas en inventario
            for order in self.inventory:
                # Encontrar posición accesible para entregar
                accessible_position = self._get_nearest_accessible_position(order.dropoff, game_map)
                
                if accessible_position:
                    distance = self._manhattan_distance(self.grid_x, self.grid_y, 
                                                    accessible_position[0], accessible_position[1])
                    
                    score = self._evaluate_order_delivery(order, current_time, game_map, weather_system, distance)
                    print(f"  Entrega {order.id} - Dropoff: {order.dropoff} -> Accesible: {accessible_position} - Score: {score:.1f}")
                    
                    if score > best_score:
                        best_score = score
                        best_action = 'deliver'
                        best_order = order
                        best_accessible_position = accessible_position
                else:
                    print(f"  Entrega {order.id} - Sin posición accesible para dropoff {order.dropoff}")
            
            if best_action and best_order and best_accessible_position:
                self.current_target = best_accessible_position  # Usar la posición accesible, no el edificio
                
                if best_action == 'pickup':
                    print(f"🎯 CPU Medium - Objetivo: Recoger {best_order.id}")
                    print(f"   Edificio pickup: {best_order.pickup}")
                    print(f"   Posición accesible: {best_accessible_position}")
                else:
                    print(f"🎯 CPU Medium - Objetivo: Entregar {best_order.id}")
                    print(f"   Edificio dropoff: {best_order.dropoff}")
                    print(f"   Posición accesible: {best_accessible_position}")
                
                # Generar camino a la posición accesible
                self._generate_direct_path(self.current_target, game_map)
            else:
                # Movimiento exploratorio
                print("CPU Medium - Sin objetivos buenos, movimiento exploratorio")
                self._exploratory_move(game_map)
                
        except Exception as e:
            print(f"❌ Error en _update_medium: {e}")
            import traceback
            traceback.print_exc()
            # En caso de error, hacer movimiento exploratorio
            self._exploratory_move(game_map)
    
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
        """Planificación de ruta óptima (nivel difícil)"""
        if not active_orders and self.inventory.is_empty():
            self._exploratory_move(game_map)
            return
        
        current_time = game_time.get_current_game_time()
        best_sequence = None
        best_score = -float('inf')
        
        # Evaluar diferentes secuencias de pedidos
        orders_to_consider = list(active_orders)[:3]  # Limitar por rendimiento
        
        for order in orders_to_consider:
            if self.can_pickup_order(order):
                sequence_score = self._evaluate_sequence([order], current_time, game_map)
                if sequence_score > best_score:
                    best_score = sequence_score
                    best_sequence = [('pickup', order)]
        
        # Evaluar entregas en inventario
        for order in self.inventory:
            sequence_score = self._evaluate_sequence([], current_time, game_map, delivery_order=order)
            if sequence_score > best_score:
                best_score = sequence_score
                best_sequence = [('deliver', order)]
        
        if best_sequence:
            action_type, order = best_sequence[0]
            self.current_target = order.pickup if action_type == 'pickup' else order.dropoff
            print(f"CPU Hard - Ruta óptima: {action_type} {order.id} (score: {best_score:.1f})")
            self._generate_astar_path(self.current_target, game_map)
        else:
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
        """Genera camino usando algoritmo A* (nivel difícil)"""
        self.path.clear()
        
        start = (self.grid_x, self.grid_y)
        goal = target
        
        # Implementación simple de A*
        open_set = {start}
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: self._manhattan_distance(start[0], start[1], goal[0], goal[1])}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                # Reconstruir camino
                self._reconstruct_path(came_from, current)
                return
            
            open_set.remove(current)
            
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Verificar validez
                if not (0 <= neighbor[0] < game_map.width and 0 <= neighbor[1] < game_map.height):
                    continue
                
                if game_map.legend.get(game_map.tiles[neighbor[1]][neighbor[0]], {}).get("blocked", False):
                    continue
                
                # Coste del movimiento
                tentative_g_score = g_score[current] + 1
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._manhattan_distance(neighbor[0], neighbor[1], goal[0], goal[1])
                    if neighbor not in open_set:
                        open_set.add(neighbor)
    
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
        """Sigue el camino generado - MEJORADO"""
        # Verificar si ya llegamos al objetivo actual
        if self.current_target:
            target_x, target_y = self.current_target
            current_distance = max(abs(self.grid_x - target_x), abs(self.grid_y - target_y))
            
            if current_distance <= 1:
                print(f"🎯 CPU está en posición objetivo o adyacente: {self.current_target}")
                self.path.clear()  # Limpiar camino ya que estamos cerca
        
        if self.path and not self.is_moving and self.move_cooldown <= 0:
            next_pos = self.path[0]
            
            # Si ya estamos en la siguiente posición, saltarla
            if (self.grid_x, self.grid_y) == next_pos:
                self.path.popleft()
                if not self.path:
                    return
                next_pos = self.path[0]
            
            dx = next_pos[0] - self.grid_x
            dy = next_pos[1] - self.grid_y
            
            # Normalizar dirección
            dx = 1 if dx > 0 else -1 if dx < 0 else 0
            dy = 1 if dy > 0 else -1 if dy < 0 else 0
            
            # Obtener multiplicadores para el movimiento
            weather_multiplier = weather_system.get_speed_multiplier()
            tile_char = game_map.tiles[self.grid_y][self.grid_x]
            surface_multiplier = game_map.legend.get(tile_char, {}).get("surface_weight", 1.0)
            
            # Intentar moverse
            if self.try_move(dx, dy, game_map.tiles, weather_multiplier, surface_multiplier):
                old_pos = (self.grid_x - dx, self.grid_y - dy)  # Posición anterior
                self.path.popleft()
                print(f"CPU se movió de {old_pos} a ({self.grid_x}, {self.grid_y})")
    
    def _exploratory_move(self, game_map):
        """Movimiento exploratorio cuando no hay objetivos claros"""
        # Buscar posición aleatoria válida
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
                    self._generate_greedy_path(self.current_target, game_map)
                break
            attempts += 1
    
# entities/cpu_player.py (modificaciones específicas)

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
        """Calcula distancia usando A* entre dos puntos"""
        # Para simplificar, usar Manhattan en esta implementación
        return self._manhattan_distance(start[0], start[1], goal[0], goal[1])
    
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
            (building_x, building_y - 1),    # Arriba
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
        """Genera un camino directo y eficiente al objetivo - NUEVO MÉTODO"""
        self.path.clear()
        
        start_x, start_y = self.grid_x, self.grid_y
        target_x, target_y = target
        
        print(f"🔄 Generando camino directo de ({start_x}, {start_y}) a ({target_x}, {target_y})")
        
        current_x, current_y = start_x, start_y
        
        # Usar algoritmo simple pero efectivo
        max_steps = 100
        steps = 0
        
        while (current_x, current_y) != (target_x, target_y) and steps < max_steps:
            # Decidir dirección preferida
            dx = 0
            dy = 0
            
            if current_x < target_x:
                dx = 1
            elif current_x > target_x:
                dx = -1
            elif current_y < target_y:
                dy = 1
            elif current_y > target_y:
                dy = -1
            
            # Intentar movimiento preferido primero
            new_x, new_y = current_x + dx, current_y + dy
            
            if (0 <= new_x < game_map.width and 0 <= new_y < game_map.height and
                not game_map.legend.get(game_map.tiles[new_y][new_x], {}).get("blocked", False)):
                
                self.path.append((new_x, new_y))
                current_x, current_y = new_x, new_y
            else:
                # Si el movimiento preferido no es posible, intentar alternativas
                moved = False
                for alt_dx, alt_dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if alt_dx == dx and alt_dy == dy:
                        continue  # Saltar el que ya intentamos
                        
                    alt_x, alt_y = current_x + alt_dx, current_y + alt_dy
                    if (0 <= alt_x < game_map.width and 0 <= alt_y < game_map.height and
                        not game_map.legend.get(game_map.tiles[alt_y][alt_x], {}).get("blocked", False)):
                        
                        self.path.append((alt_x, alt_y))
                        current_x, current_y = alt_x, alt_y
                        moved = True
                        break
                
                if not moved:
                    break  # No hay movimientos posibles
            
            steps += 1
        
        print(f"✅ Camino generado: {len(self.path)} pasos")
        if self.path:
            print(f"   Primer paso: {self.path[0]}, Último paso: {self.path[-1]}")