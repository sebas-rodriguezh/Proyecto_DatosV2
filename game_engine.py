import pygame
from datetime import datetime
from entities.player import Player
from ui.map import Map
from api.api_manager import APIManager
from entities.weather import Weather
from core.game_time import GameTime
from entities.order_list import OrderList
from entities.order import Order
from ui.ui_manager import UIManager
from utils.interaction_manager import InteractionManager
from core.game_state import GameState
from utils.undo_stack import UndoRedoManager
from utils.setup_directories import setup_directories
import json
import os
from datetime import datetime
from utils.save_load_manager import SaveLoadManager
from datetime import timedelta
from ui.order_popup_manager import OrderPopupManager
from utils.score_manager import score_manager

    
class GameEngine:
    """Motor principal del juego que coordina todos los sistemas"""
    
    def __init__(self, load_slot=None, cpu_difficulty=None):
        pygame.init()
        try:
            pygame.font.init()
            test_font = pygame.font.Font(None, 16)
        except:
            print("Error inicializando fuentes de Pygame")
        from utils.setup_directories import setup_directories
        setup_directories()
        
        from utils.score_manager import score_manager
        score_manager.initialize_score_system()
        
        # Configuración inicial
        self.api = APIManager()
        self.setup_game_data()
        
        # Crear sistemas principales
        self.game_state = GameState()
        self.setup_display()

        self.popup_manager = OrderPopupManager(self.screen_width, self.screen_height)

        # Sistema de guardado/carga
        self.save_manager = SaveLoadManager()
        
        if load_slot:
            self.load_game(load_slot)
        else:
            self.setup_game_objects()
            
        self.cpu_player = None
        if cpu_difficulty and cpu_difficulty in ["easy", "medium", "hard"]:
            self.setup_cpu_player(cpu_difficulty)
        
        self.setup_managers()
        from ui.pause_menu import PauseMenu
        self.pause_menu = PauseMenu(self.screen, self.save_manager)
        
        self.running = True
        self.clock = pygame.time.Clock()
        self.last_time = pygame.time.get_ticks()
        


    def setup_cpu_player(self, difficulty):
        """Configura el jugador CPU"""
        from cpu_player import CPUPlayer
        
        # Posición inicial diferente al jugador humano
        cpu_x = max(0, self.cols // 2 - 3)
        cpu_y = max(0, self.rows // 2 - 3)
        
        self.cpu_player = CPUPlayer(
            cpu_x, cpu_y, 
            self.game_map.tile_size, 
            self.game_map.legend,
            difficulty=difficulty
        )
        
        print(f"Jugador CPU inicializado (dificultad: {difficulty}) en posición ({cpu_x}, {cpu_y})")
        

    def setup_game_data(self):
        """Carga datos iniciales de la API o caché local - VERSIÓN SIMPLIFICADA"""
        try:
            self.map_data = self.api.get_map_data()
            self.jobs_data = self.api.get_jobs()
            self.weather_data = self.api.get_weather()
            print("Datos cargados correctamente")
        except Exception as e:
            print(f"Error crítico: No se pudieron cargar los datos: {e}")
            raise Exception("No se pueden cargar datos y no hay respaldo disponible")


    def load_default_data(self):
        """Carga datos por defecto desde archivos locales"""
        try:
            import json
            import os
            
            # Cargar mapa por defecto
            with open('data/map_data.json', 'r', encoding='utf-8') as f:
                self.map_data = json.load(f)
            
            # Cargar trabajos por defecto
            with open('data/jobs_data.json', 'r', encoding='utf-8') as f:
                self.jobs_data = json.load(f)
            
            # Cargar clima por defecto
            with open('data/weather_data.json', 'r', encoding='utf-8') as f:
                self.weather_data = json.load(f)
                
            print("Datos por defecto cargados correctamente")
            
        except Exception as e:
            print(f"Error crítico: No se pudieron cargar los datos por defecto: {e}")
            raise
    
    def setup_display(self):
        """Configura la pantalla y elementos visuales"""
        self.game_map = Map(self.map_data, tile_size=20)
        self.rows, self.cols = self.game_map.height, self.game_map.width
        self.screen_width = self.cols * self.game_map.tile_size + 300
        self.screen_height = self.rows * self.game_map.tile_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Courier Quest")
    
    def setup_game_objects(self, loaded_data=None):
        """Crea los objetos principales del juego"""
        if loaded_data:
            # Cargar desde datos guardados
            self.load_from_save_data(loaded_data)
            self.verify_order_consistency()
        else:
            # Configuración inicial nueva partidal
            self.setup_new_game()
    
    def setup_new_game(self):
        """Configura una nueva partida desde cero"""
        game_start_datetime = self.get_game_start_time_from_json()

        self.all_orders = OrderList.from_api_response(self.jobs_data)  # Todos los pedidos
        self.active_orders = OrderList.create_empty()  # Pedidos activos (liberados)
        self.pending_orders = OrderList.create_empty()  # Pedidos pendientes de liberar
        self.completed_orders = OrderList.create_empty()
        self.rejected_orders = OrderList.create_empty()  # Pedidos rechazados

        print("DEBUG RELEASE_TIMES:")
        for i, order in enumerate(self.all_orders):
            print(f"   Pedido {i+1}: {order.id} - release_time: {order.release_time}s")
            if order.release_time == 0:
                self.active_orders.enqueue(order)  
            else:
                self.pending_orders.enqueue(order)  
        
        # Crear jugador
        self.player = Player(self.cols // 2, self.rows // 2, 
                        self.game_map.tile_size, self.game_map.legend)

        # Crear tiempo de juego 
        self.game_time = GameTime(
            total_duration_min=15,
            game_start_time=game_start_datetime,  # Hora del JSON
            time_scale=6.0  # ← ESCALA TEMPORAL (Modificar el parámetro si quiere correrlo 1s real = xs juego)
        )
        self.game_time.start()

        self.weather_system = Weather(self.api)
        
        self.income_goal = self.map_data.get("goal", 1500)
        self.game_state.set_income_goal(self.income_goal)
        self.game_state.set_player_reference(self.player)

        self.undo_manager = UndoRedoManager(max_states=10)
        self.undo_manager.save_game_state(self, force=True)
        
        self.verify_order_deadlines()
        

    def _create_order_from_save_data(self, order_data):
        """Crea una orden desde datos de guardado"""
        from datetime import datetime
        
        
        deadline = order_data["deadline"]
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline)
        
        # Crear orden
        order = Order(
            id=order_data["id"],
            pickup=order_data["pickup"],
            dropoff=order_data["dropoff"],
            payout=order_data["payout"],
            deadline=deadline,
            weight=order_data["weight"],
            priority=order_data["priority"],
            release_time=order_data["release_time"]
        )
        
        if "color" in order_data:
            order.color = tuple(order_data["color"])
        
        return order
    
    def update_order_expirations(self):
        """Actualiza el estado de expiración de todos los pedidos activos - CORREGIDO"""
        current_game_time = self.game_time.get_current_game_time()
        
        expired_orders = []
        
        # 1. Pedidos activos
        for order in list(self.active_orders): 
            if not order.is_in_inventory and not order.is_completed:
                if order.check_expiration(current_game_time):
                    expired_orders.append(order)
                    print(f"Pedido {order.id} EXPIRADO (no recogido) a las {current_game_time.strftime('%H:%M:%S')}")
        
        # 2. Pedidos en inventario
        for order in list(self.player.inventory):
            if order.is_in_inventory and not order.is_completed:
                if order.check_expiration(current_game_time):
                    expired_orders.append(order)
                    print(f"Pedido {order.id} EXPIRADO (no entregado) a las {current_game_time.strftime('%H:%M:%S')}")
        
        # Procesar pedidos expirados
        for expired_order in expired_orders:
            self.handle_expired_order(expired_order, current_game_time)

    def handle_expired_order(self, order, current_time):
        """Maneja las consecuencias de un pedido expirado"""
        reputation_penalty = 6
        old_reputation = self.player.reputation
        self.player.reputation = max(0, self.player.reputation - reputation_penalty)
        
        self.game_state.orders_cancelled += 1
        self.game_state.current_streak = 0  
        
        if order.is_in_inventory:
            self.player.remove_from_inventory(order.id)
            location = "inventario"
        else:
            self.active_orders.remove_by_id(order.id)
            location = "activos"
        
        message = f"{order.id} EXPIRADO (-{reputation_penalty} reputación)"
        self.ui_manager.show_message(message, 5)
        
        print(f"Pedido {order.id} expirado ({location}). Reputación: {old_reputation} → {self.player.reputation}")
        print(f"Deadline: {order.deadline.strftime('%H:%M:%S')}, Hora actual: {current_time.strftime('%H:%M:%S')}")

    def verify_order_deadlines(self):
        """Verifica que todos los deadlines sean correctos"""
        print(f"Hora inicio juego: {self.game_time.game_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_orders = list(self.all_orders) if hasattr(self, 'all_orders') else []
        
        print(f"\nTODOS LOS PEDIDOS ({len(all_orders)}):")
        for order in all_orders:
            time_remaining = order.get_time_remaining(self.game_time.get_current_game_time())
            minutes = time_remaining / 60
            print(f"   {order.id}: {time_remaining:.0f}s ({minutes:.1f} min) - {order.deadline.strftime('%Y-%m-%d %H:%M:%S')}")
        
        expected_date = self.game_time.game_start_time.date()
        inconsistent_orders = []
        
        for order in all_orders:
            if order.deadline.date() != expected_date:
                inconsistent_orders.append(order)
        
        if inconsistent_orders:
            print(f"\nPEDIDOS CON FECHA INCONSISTENTE:")
            for order in inconsistent_orders:
                print(f"   {order.id}: {order.deadline.strftime('%Y-%m-%d')} (esperado: {expected_date})")
        else:
            print(f"\nTodos los Pedidos tienen la fecha de: {expected_date}")

    def get_game_start_time_from_json(self):
        """Extrae la hora de inicio desde map_data - VERSIÓN CORREGIDA CON DEBUG"""
        try:
            if "start_time" in self.map_data.get("data", {}):
                start_time_str = self.map_data["data"]["start_time"]
                
                if start_time_str.endswith('Z'):
                    start_time_str = start_time_str[:-1]
                
                # Asegurar formato completo
                if len(start_time_str) == 16:  
                    start_time_str += ":00"
                
                start_time = datetime.fromisoformat(start_time_str)
                return start_time
            
            if self.jobs_data and len(self.jobs_data.get('data', [])) > 0:
                first_job = self.jobs_data['data'][0]
                if "deadline" in first_job:
                    deadline_str = first_job["deadline"]                    
                    if deadline_str.endswith('Z'):
                        deadline_str = deadline_str[:-1]
                    
                    if len(deadline_str) == 16:
                        deadline_str += ":00"
                    
                    deadline = datetime.fromisoformat(deadline_str)
                    start_time = deadline - timedelta(minutes=10)
                    return start_time
                        
        except Exception as e:
            print(f"ERROR leyendo hora del JSON: {e}")
            import traceback
            traceback.print_exc()
        
        default_time = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        return default_time
    
    def load_from_save_data(self, save_data):
        """Carga el estado del juego desde datos guardados - VERSIÓN COMPLETA MEJORADA"""
        try:
            print(f"📄 Cargando partida desde datos guardados...")
            
            self.active_orders = OrderList.create_empty()
            self.completed_orders = OrderList.create_empty()
            self.pending_orders = OrderList.create_empty()
            self.rejected_orders = OrderList.create_empty()
            
            if "all_orders" in save_data:
                self.all_orders = OrderList.create_empty()
                for order_data in save_data["all_orders"]:
                    try:
                        order = self._create_order_from_save_data(order_data)
                        self.all_orders.enqueue(order)
                    except Exception as e:
                        pass
                print(f"{len(self.all_orders)} pedidos base cargados")
            else:
                self.all_orders = OrderList.from_api_response(self.jobs_data)
                print(f"No hay all_orders en guardado, recreando desde API: {len(self.all_orders)} pedidos")
            
            
            for order_data in save_data["active_orders"]:
                try:
                    order = self._find_or_create_order(order_data, self.all_orders)
                    if order:
                        self.active_orders.enqueue(order)
                except Exception as e:
                    print(f"Error cargando orden activa: {e}")
            
            if "pending_orders" in save_data:
                for order_data in save_data["pending_orders"]:
                    try:
                        order = self._find_or_create_order(order_data, self.all_orders)
                        if order:
                            self.pending_orders.enqueue(order)
                    except Exception as e:
                        print(f"Error cargando orden pendiente: {e}")
            
            # Cargar órdenes completadas
            for order_data in save_data["completed_orders"]:
                try:
                    order = self._find_or_create_order(order_data, self.all_orders)
                    if order:
                        order.mark_as_completed()
                        self.completed_orders.enqueue(order)
                except Exception as e:
                    print(f"Error cargando orden completada: {e}")
            
            # Cargar órdenes rechazadas
            if "rejected_orders" in save_data:
                for order_data in save_data["rejected_orders"]:
                    try:
                        order = self._find_or_create_order(order_data, self.all_orders)
                        if order:
                            self.rejected_orders.enqueue(order)
                    except Exception as e:
                        print(f"Error cargando orden rechazada: {e}")
            

            player_data = save_data["player_data"]
            self.player = Player(
                player_data["grid_x"],
                player_data["grid_y"], 
                self.game_map.tile_size, 
                self.game_map.legend
            )
            
            # Restaurar propiedades del jugador
            self.player.stamina = player_data["stamina"]
            self.player.reputation = player_data["reputation"]
            self.player.current_weight = player_data["current_weight"]
            self.player.state = player_data["state"]
            self.player.direction = player_data["direction"]
            
            # Limpiar inventarios existentes
            self.player.inventory.clear()
            self.player.completed_orders.clear()
            
            for order_data in player_data["inventory"]:
                try:
                    order = self._find_or_create_order(order_data, self.all_orders)
                    if order:
                        order.is_in_inventory = True
                        order.is_expired = order_data.get("is_expired", False)
                        
                        self.player.inventory.enqueue(order)
                        print(f"{order.id} añadido al inventario")
                except Exception as e:
                    print(f"Error cargando orden al inventario: {e}")
            
            for order_data in player_data["completed_orders"]:
                try:
                    order = self._find_or_create_order(order_data, self.all_orders)
                    if order:
                        order.mark_as_completed()
                        self.player.completed_orders.enqueue(order)
                except Exception as e:
                    print(f"Error cargando orden completada del jugador: {e}")
            
            game_state_data = save_data["game_state"]
            self.game_state.total_earnings = game_state_data["total_earnings"]
            self.game_state.income_goal = game_state_data["income_goal"]
            self.game_state.game_over = game_state_data["game_over"]
            self.game_state.victory = game_state_data["victory"]
            self.game_state.game_over_reason = game_state_data["game_over_reason"]
            self.game_state.orders_completed = game_state_data["orders_completed"]
            self.game_state.orders_cancelled = game_state_data["orders_cancelled"]
            self.game_state.perfect_deliveries = game_state_data["perfect_deliveries"]
            self.game_state.late_deliveries = game_state_data["late_deliveries"]
            self.game_state.current_streak = game_state_data["current_streak"]
            self.game_state.best_streak = game_state_data["best_streak"]
            
            start_time_str = game_state_data["start_time"]
            if isinstance(start_time_str, str):
                self.game_state.start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            
            self.game_state.set_player_reference(self.player)
            
            game_time_data = save_data["game_time"]
            elapsed_time = game_time_data.get("elapsed_time_sec", 0)
            total_duration = game_time_data.get("total_duration", 900)
            time_scale = game_time_data.get("time_scale", 3.0)
            
            if "game_start_time" in save_data:
                game_start_time_str = save_data["game_start_time"]
                if isinstance(game_start_time_str, str):
                    game_start_time = datetime.fromisoformat(game_start_time_str.replace('Z', '+00:00'))
                else:
                    game_start_time = self.get_game_start_time_from_json()
            else:
                game_start_time = self.get_game_start_time_from_json()
            
            self.game_time = GameTime(
                total_duration_min=total_duration/60,
                game_start_time=game_start_time,
                time_scale=time_scale
            )
            
            
            import pygame
            current_pygame_time = pygame.time.get_ticks() / 1000.0
            
            
            if "pygame_start_time" in game_time_data and "start_real_time" in game_time_data:
                
                self.game_time.pygame_start_time = current_pygame_time - elapsed_time
                self.game_time.start_real_time = 0  
            else:
                
                self.game_time.pygame_start_time = current_pygame_time - elapsed_time
                self.game_time.start_real_time = 0


            self.game_time.paused = False
            self.game_time.pause_duration = 0
                        
            print(f"Tiempo restaurado: {elapsed_time:.1f}s transcurridos de {total_duration}s totales")
            
            weather_data = save_data["weather_state"]
            self.weather_system = Weather(self.api)
            
            # Configurar clima actual
            try:
                from entities.weather import WeatherCondition
                condition_str = weather_data["current_condition"]
                # Buscar la condición climática correspondiente
                for condition in WeatherCondition:
                    if condition.value == condition_str:
                        self.weather_system.current_condition = condition
                        break
                
                self.weather_system.current_intensity = weather_data.get("current_intensity", 0.0)
                self.weather_system.current_multiplier = weather_data.get("current_multiplier", 1.0)
            except Exception as e:
                print(f"Error configurando clima: {e}")
            
            self.camera_x, self.camera_y = save_data["camera_position"]
            self.income_goal = save_data["income_goal"]
            
            self.undo_manager = UndoRedoManager(max_states=10)
            self.undo_manager.save_game_state(self, force=True)
            
            self.setup_managers()
            
            print("Partida cargada correctamente")
            print(f"   - Jugador en posición: ({self.player.grid_x}, {self.player.grid_y})")
            print(f"   - Órdenes activas: {len(self.active_orders)}")
            print(f"   - Órdenes en inventario: {len(self.player.inventory)}")
            print(f"   - Ganancias: ${self.game_state.total_earnings}")
            print(f"   - Tiempo restante: {self.game_time.get_remaining_time_formatted()}")
            
            self.verify_order_consistency()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.setup_new_game()

    def _find_or_create_order(self, order_data, all_orders):
        """Busca una orden en all_orders o la crea si no existe - MEJORADO"""
        existing_order = all_orders.find_by_id(order_data["id"])
        if existing_order:
            existing_order.is_expired = order_data.get("is_expired", False)
            existing_order.is_completed = order_data.get("is_completed", False)
            existing_order.is_in_inventory = order_data.get("is_in_inventory", False)
            
            if order_data.get("accepted_time"):
                try:
                    existing_order.accepted_time = datetime.fromisoformat(
                        order_data["accepted_time"].replace('Z', '+00:00')
                    )
                except:
                    pass
            
            if "color" in order_data:
                existing_order.color = tuple(order_data["color"])
                    
            return existing_order
        else:
            return self._create_order_from_save_data(order_data)

    def verify_order_consistency(self):
        """Verifica la consistencia de las órdenes después de cargar - MEJORADO"""
        
        all_order_ids = set()
        duplicates = []
        
        order_lists = [
            ("active_orders", self.active_orders),
            ("pending_orders", self.pending_orders), 
            ("completed_orders", self.completed_orders),
            ("rejected_orders", self.rejected_orders),
            ("inventory", self.player.inventory),
            ("all_orders", self.all_orders)
        ]
        
        for order_list_name, order_list in order_lists:
            for order in order_list:
                if order.id in all_order_ids:
                    duplicates.append((order.id, order_list_name))
                all_order_ids.add(order.id)
        
        if duplicates:
            for order_id, list_name in duplicates:
                print(f"   - {order_id} en {list_name}")
        else:
            pass
        
        calculated_weight = sum(order.weight for order in self.player.inventory)
        if self.player.current_weight != calculated_weight:
            print(f"Peso inconsistente: {self.player.current_weight} vs {calculated_weight}")
            self.player.current_weight = calculated_weight
            print(f"Peso corregido a: {self.player.current_weight}kg")
        else:
            print(f"Peso del inventario consistente: {self.player.current_weight}kg")
        
        inventory_issues = 0
        for order in self.player.inventory:
            if not order.is_in_inventory:
                print(f"Orden {order.id} en inventario pero is_in_inventory=False")
                order.is_in_inventory = True
                inventory_issues += 1
        
        if inventory_issues > 0:
            print(f"Corregidos {inventory_issues} estados de inventario")
        

    def _create_order_from_save_data(self, order_data):
        """Crea una orden desde datos de guardado"""

        
        deadline = order_data["deadline"]
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline)
        
        # Crear orden
        order = Order(
            id=order_data["id"],
            pickup=order_data["pickup"],
            dropoff=order_data["dropoff"],
            payout=order_data["payout"],
            deadline=deadline,
            weight=order_data["weight"],
            priority=order_data["priority"],
            release_time=order_data["release_time"]
        )
        
        if "color" in order_data:
            order.color = tuple(order_data["color"])
        
        return order
    
    def save_game(self, slot_name="slot1"):
        """Guarda el estado actual del juego"""
        self.verify_order_consistency()
        
        success = self.save_manager.save_game(self, slot_name)
        if success:
            print(f"Partida guardada en slot: {slot_name}")
            return True
        else:
            print("Error al guardar partida")
            return False
        
    def load_game(self, slot_name="slot1"):
        """Carga una partida guardada"""
        save_data = self.save_manager.load_game(slot_name)
        if save_data:
            self.setup_game_objects(save_data)
            return True
        else:
            print(f"No se encontró partida en slot: {slot_name}")
            print("Iniciando nueva partida...")
            self.setup_game_objects()
            return False
    
    def update_release_times(self, dt):
        """Libera pedidos según su release_time"""
        current_game_time_elapsed = self.game_time.get_elapsed_game_time()  
        
        orders_to_release = []
        remaining_orders = OrderList.create_empty()
        
        while not self.pending_orders.is_empty():
            order = self.pending_orders.dequeue()
            
            if current_game_time_elapsed  >= order.release_time:
                orders_to_release.append(order)
                self.ui_manager.show_message(f"Nuevo pedido: {order.id}", 3)
            else:
                remaining_orders.enqueue(order)
        
        self.pending_orders = remaining_orders

        for order in orders_to_release:
            if not self.popup_manager.is_popup_active():  # Solo si no hay popup activo
                self.popup_manager.show_new_order_popup(order)
                break  
            else:
                order.release_time = current_game_time_elapsed  + 5
                self.pending_orders.enqueue(order)
          
    
    def setup_managers(self):
        """Configura los managers del juego"""
        self.ui_manager = UIManager(self.screen, self.game_map, self.screen_width, self.screen_height)
        self.interaction_manager = InteractionManager(self.player, self.active_orders, self.completed_orders, self.game_time)
        
        self.ui_manager.interaction_manager = self.interaction_manager
        
        self.popup_manager.game_engine = self
        
        self.camera_x, self.camera_y = 0, 0


    def handle_events(self):
        """Maneja todos los eventos del juego - VERSIÓN CORREGIDA"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            if event.type == pygame.KEYDOWN:
                # Tecla D para debug de CPU
                if event.key == pygame.K_l and self.cpu_player:
                    self.debug_cpu_status()
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if not self.game_state.game_over:
                    if self.popup_manager.popup_active:
                        self.popup_manager.popup_active = False
                        self.popup_manager.pending_order = None
                    elif self.popup_manager.cancel_popup_active:
                        self.popup_manager.cancel_popup_active = False
                        self.popup_manager.selected_order_for_cancel = None
                    else:
                        self.pause_menu.toggle()
                continue  
            
            if self.pause_menu.active:
                pause_result = self.pause_menu.handle_event(event)
                if pause_result:
                    self.handle_pause_result(pause_result)
                continue
            
            popup_result = self.popup_manager.handle_event(event, self, self.player, self.active_orders)
            if popup_result:
                message = popup_result.get("message", "")
                if message:
                    self.ui_manager.show_message(message, 3)

                penalty = popup_result.get("penalty")
                if penalty:
                    old_reputation = self.player.reputation
                    self.player.reputation = max(0, self.player.reputation - penalty)
                    

                    if (popup_result.get("type") == "cancel_order" and 
                        popup_result.get("result") == "confirmed"):
                        print(f"Cancelación confirmada - Total cancelaciones: {self.game_state.orders_cancelled}")        

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:  # Click derecho
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.handle_inventory_right_click(mouse_x, mouse_y)            

            # Manejo de undo/redo
            if event.type == pygame.KEYDOWN:
                # Undo con Ctrl+Z
                if event.key == pygame.K_z and pygame.key.get_pressed()[pygame.K_LCTRL]:
                    if self.undo_manager.undo_last_action(self):
                        self.ui_manager.show_message("Acción deshecha", 2)
                    else:
                        self.ui_manager.show_message("No se puede deshacer", 2)
                
                # Redo con Ctrl+Y
                elif event.key == pygame.K_y and pygame.key.get_pressed()[pygame.K_LCTRL]:
                    if self.undo_manager.redo_last_action(self):
                        self.ui_manager.show_message("Acción rehecha", 2)
                    else:
                        self.ui_manager.show_message("No se puede rehacer", 2)
                
                # Guardar partida con Ctrl+S
                elif event.key == pygame.K_s and pygame.key.get_pressed()[pygame.K_LCTRL]:
                    if self.save_game("slot1"):
                        self.ui_manager.show_message("Partida guardada", 2)
                    else:
                        self.ui_manager.show_message("Error al guardar", 2)
                
                # Cargar partida con Ctrl+L
                elif event.key == pygame.K_l and pygame.key.get_pressed()[pygame.K_LCTRL]:
                    if self.load_game("slot1"):
                        self.ui_manager.show_message("Partida cargada", 2)
                    else:
                        self.ui_manager.show_message("Error al cargar", 2)
                
                if event.key == pygame.K_p:  # Tecla P para prioridad
                    self.player.reorganize_inventory_by_priority()
                    self.ui_manager.show_message("Inventario ordenado por PRIORIDAD", 2)
                    
                elif event.key == pygame.K_o:  # Tecla O para deadline
                    self.player.reorganize_inventory_by_deadline()
                    self.ui_manager.show_message("Inventario ordenado por URGENCIA", 2)                
            
            # Delegar eventos a los managers apropiados
            self.ui_manager.handle_event(event, self.active_orders, self.player)
            
            if not self.game_state.game_over and not self.pause_menu.active:
                self.interaction_manager.handle_event(event, self.game_state, self.game_map)
            
            # Guardar estado después de interacciones importantes
            if not self.game_state.game_over and not self.pause_menu.active:
                self.undo_manager.save_game_state(self)   

    def handle_pause_result(self, result):
        """Maneja las acciones del menú de pausa"""
        action = result.get("action")
        
        if action == "resume":
            self.pause_menu.toggle()
        elif action == "confirm_save":
            slot_name = result.get("slot")
            if self.save_game(slot_name):
                self.ui_manager.show_message(f"Partida guardada en {slot_name}", 3)
            self.pause_menu.toggle()
        elif action == "main_menu":
            self.running = False

    def handle_inventory_right_click(self, mouse_x, mouse_y):
        """Maneja cancelación de pedidos desde el inventario con click derecho"""
        cols = self.game_map.width
        x_offset = cols * self.game_map.tile_size
        
        if mouse_x > x_offset:  # Está en el sidebar
            # Calcular qué pedido del inventario fue clickeado
            inventory_start_y = 235
            item_height = 45
            
            if mouse_y >= inventory_start_y and self.player.inventory:
                item_index = (mouse_y - inventory_start_y) // item_height
                
                if 0 <= item_index < len(self.player.inventory):
                    order = self.player.inventory[item_index]
                    self.popup_manager.show_cancel_order_popup(order)


    def update(self, dt):
        """Actualiza todos los sistemas del juego - MODIFICADO"""
        if not self.game_state.game_over:
            self.game_time.update(dt)
            self.weather_system.update(dt)
            self.update_release_times(dt)
            
            self.update_order_expirations()
            
            self.update_player_movement(dt)
            
            # Actualizar jugador CPU si existe - ESTA ES LA PARTE IMPORTANTE
            if self.cpu_player:
                self.cpu_player.update(dt, self.active_orders, self.game_map, 
                                     self.game_time, self.weather_system)
                # Las interacciones se manejan dentro de update() de CPUPlayer
            
            self.interaction_manager.update(dt)
            
            self.update_game_state()
            self.popup_manager.update(dt)
            
            if self.player.is_moving:
                self.undo_manager.save_game_state(self)
        
        self.update_camera()
    


    def update_player_movement(self, dt):
        """Actualiza el movimiento del jugador - VERSIÓN MODIFICADA"""
        
        weather_multiplier = self.weather_system.get_speed_multiplier()
        weather_stamina_consumption = self.weather_system.get_stamina_consumption()
        
        self.player.update_movement(dt, weather_stamina_consumption)
        
        if not self.player.is_moving:
            keys = pygame.key.get_pressed()
            dx, dy = 0, 0
            
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                dx = -1
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                dx = 1
            elif keys[pygame.K_UP] or keys[pygame.K_w]:
                dy = -1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                dy = 1
            
            if dx != 0 or dy != 0:
                tile_x, tile_y = self.player.grid_x, self.player.grid_y
                if 0 <= tile_y < len(self.game_map.tiles) and 0 <= tile_x < len(self.game_map.tiles[0]):
                    tile_char = self.game_map.tiles[tile_y][tile_x]
                    surface_multiplier = self.game_map.legend[tile_char].get("surface_weight", 1.0)
                else:
                    surface_multiplier = 1.0

                self.player.try_move(dx, dy, self.game_map.tiles, weather_multiplier, surface_multiplier)
    
    def update_camera(self):
        """Actualiza la posición de la cámara - USAR POSICIÓN VISUAL"""
        visual_x, visual_y = self.player.get_visual_position()
        
        self.camera_x = visual_x * self.game_map.tile_size - self.screen_width // 2 + 150
        self.camera_y = visual_y * self.game_map.tile_size - self.screen_height // 2
        self.camera_x = max(0, min(self.camera_x, self.cols * self.game_map.tile_size - self.screen_width + 300))
        self.camera_y = max(0, min(self.camera_y, self.rows * self.game_map.tile_size - self.screen_height))
    
    def update_game_state(self):
        """Actualiza el estado general del juego - VERSIÓN MEJORADA"""
        if self.game_state.game_over:
            return
        
        if self.player.reputation < 20:
            print("Fin del juego: Reputación muy baja")
            self.game_state.set_game_over(False, "Derrota: Reputación muy baja")
            self.save_final_score(False)
            return
        
        if self.game_time.is_time_up() and self.game_state.total_earnings < self.income_goal:
            print("Fin del juego: Tiempo agotado")
            self.game_state.set_game_over(False, "Derrota: Tiempo agotado")
            self.save_final_score(False)
            return
        
        if self.game_state.total_earnings >= self.income_goal:
            print("Fin del juego: Victoria alcanzada")
            self.game_state.set_game_over(True, "¡Victoria! Meta alcanzada")
            self.save_final_score(True)
            return
        
        if self.no_more_available_orders() and self.game_state.total_earnings <= self.income_goal:
            self.game_state.set_game_over(False, "Derrota: No quedan pedidos y no se alcanzó la meta")
            self.save_final_score(False)
            return
    
        if self.no_more_available_orders():
            print("Fin del juego: No quedan pedidos disponibles")
            self.game_state.set_game_over(False, "Derrota: No quedan pedidos disponibles")
            self.save_final_score(False)
            return
  
    def no_more_available_orders(self):
        """Verifica si no quedan pedidos disponibles para completar - VERSIÓN CORREGIDA"""
        
        no_active_orders = (
            len(self.active_orders) == 0 and 
            len(self.pending_orders) == 0 and 
            len(self.player.inventory) == 0 and
            not self.popup_manager.has_pending_order()
        )
        
        if not no_active_orders:
            return False
        

        unique_cancellations = max(0, self.game_state.orders_cancelled - len(self.rejected_orders))
        
        total_processed = (
            len(self.completed_orders) + 
            len(self.rejected_orders) +
            unique_cancellations  
        )
        
        expired_count = sum(1 for order in self.all_orders if order.is_expired and not order.is_completed)
        total_processed += expired_count
        
        all_orders_processed = (total_processed >= len(self.all_orders))
        
        # DEBUG: Para verificar qué está pasando
        print(f"VERIFICACIÓN FIN DEL JUEGO:")
        print(f"   - Completados: {len(self.completed_orders)}")
        print(f"   - Rechazados: {len(self.rejected_orders)}")
        print(f"   - Cancelados (inventario): {self.game_state.orders_cancelled}")
        print(f"   - Cancelados únicos: {unique_cancellations}")
        print(f"   - Expirados: {expired_count}")
        print(f"   - Popup activo: {self.popup_manager.has_pending_order()}")
        print(f"   - Total procesado: {total_processed}")
        print(f"   - Total pedidos: {len(self.all_orders)}")
        print(f"   - ¿Juego debe terminar? {all_orders_processed}")

        return all_orders_processed



    def save_final_score(self, victory: bool):
        """Guarda la puntuación final - VERSIÓN CORREGIDA"""
        try:
            if not self.game_state.game_over:
                print("El juego no ha terminado, no se puede guardar puntuación")
                return
                
            from utils.score_manager import score_manager
            
            if not score_manager.initialized:
                score_manager.initialize_score_system()
            
            # USAR TIEMPO DE JUEGO REAL
            game_duration = self.game_time.get_elapsed_game_time()
            total_game_duration = 15 * 60  # 15 minutos en segundos
            

            final_score = self.game_state.calculate_final_score(game_duration, total_game_duration)
            
            stats = self.game_state.get_game_stats(game_duration)
            
            print(f"VERIFICACIÓN DETALLADA DE CANCELACIONES:")
            print(f"   - orders_cancelled: {self.game_state.orders_cancelled}")
            print(f"   - rejected_orders count: {len(self.rejected_orders) if hasattr(self, 'rejected_orders') else 'N/A'}")

            rejected_count = len(self.rejected_orders) if hasattr(self, 'rejected_orders') else 0
            if rejected_count > 0 and self.game_state.orders_cancelled == 0:
                self.game_state.orders_cancelled = rejected_count
            
            if self.game_state.orders_cancelled > 0:
                penalty_amount = self.game_state.orders_cancelled * 100
                print(f"Penalización por {self.game_state.orders_cancelled} cancelaciones: -${penalty_amount}")
            else:
                print(f"Sin cancelaciones - Sin penalización")

            if self.game_state.orders_cancelled > 0:
                penalty_amount = self.game_state.orders_cancelled * 100
                print(f"Penalización por cancelaciones: -${penalty_amount}")
            else:
                print(f"Sin cancelaciones - Sin penalización")


            
            success = score_manager.add_score(
                self.game_state, 
                victory, 
                game_duration,
                total_game_duration
            )
            
            if success:
                pass
            else:
                print("Error al guardar puntuación en el sistema principal")
                    
        except Exception as e:
            print(f"Error guardando puntuación final: {e}")
            import traceback
            traceback.print_exc()

    def render(self):
        """Renderiza todos los elementos del juego - MODIFICADO"""
        self.screen.fill((255, 255, 255))
        
        # Dibujar juego normal
        self.render_map()
        self.weather_system.draw_particles(self.screen, self.camera_x, self.camera_y)
        self.ui_manager.draw_order_markers(self.active_orders, self.player, self.camera_x, self.camera_y, self.cpu_player)
        self.player.draw(self.screen, self.camera_x, self.camera_y)
        
        # Dibujar jugador CPU si existe
        if self.cpu_player:
            self.cpu_player.draw(self.screen, self.camera_x, self.camera_y)
        
        pending_count = len(self.pending_orders)
        
        # Pasar el cpu_player al UI manager
        self.ui_manager.draw_sidebar(self.player, self.active_orders, self.weather_system, 
                                self.game_time, self.game_state, pending_count, self.cpu_player)
        
        self.ui_manager.draw_messages()
        self.ui_manager.draw_interaction_hints(self.player, self.active_orders, self.camera_x, self.camera_y, self.game_map, self.cpu_player)        

        if self.game_state.game_over:
            self.ui_manager.draw_game_over_screen(self.game_state)

        self.popup_manager.draw_new_order_popup(self.screen)
        self.popup_manager.draw_cancel_order_popup(self.screen)
        
        if self.pause_menu.active:
            self.pause_menu.draw()
        
        pygame.display.flip()
        
    def render_map(self):
        """Renderiza el mapa del juego"""
        for y, row in enumerate(self.game_map.tiles):
            for x, char in enumerate(row):
                color = self.game_map.COLORS.get(char, (100, 100, 255))
                rect = pygame.Rect(x * self.game_map.tile_size - self.camera_x, 
                                 y * self.game_map.tile_size - self.camera_y, 
                                 self.game_map.tile_size, self.game_map.tile_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
    
    def run(self):
        """Bucle principal del juego"""
        while self.running:
            current_time = pygame.time.get_ticks()
            dt = (current_time - self.last_time) / 1000.0
            self.last_time = current_time
            
            self.handle_events()
            self.update(dt)
            self.render()
            
            self.clock.tick(60)
        
        pygame.quit()
        
    def verify_order_consistency(self):
        """Verifica la consistencia de las órdenes después de cargar"""

        all_order_ids = set()
        
        for order_list_name, order_list in [
            ("active_orders", self.active_orders),
            ("pending_orders", self.pending_orders), 
            ("completed_orders", self.completed_orders),
            ("rejected_orders", self.rejected_orders),
            ("inventory", self.player.inventory)
        ]:
            for order in order_list:
                if order.id in all_order_ids:
                    print(f"ORDEN DUPLICADA: {order.id} en {order_list_name}")
                all_order_ids.add(order.id)
        
        # Verificar peso del inventario
        calculated_weight = sum(order.weight for order in self.player.inventory)
        if self.player.current_weight != calculated_weight:
            print(f"Peso inconsistente: {self.player.current_weight} vs {calculated_weight}")
            self.player.current_weight = calculated_weight
        
        print(f"Verificación completada. {len(all_order_ids)} órdenes únicas encontradas")
        
# game_engine.py (agregar método de debug)

    def debug_cpu_status(self):
        """Muestra estado de debug de la CPU - MEJORADO PARA EDIFICIOS"""
        if self.cpu_player:
            print(f"=== DEBUG CPU ===")
            print(f"Posición CPU: ({self.cpu_player.grid_x}, {self.cpu_player.grid_y})")
            print(f"Objetivo actual: {self.cpu_player.current_target}")
            print(f"Camino restante: {len(self.cpu_player.path)} pasos")
            print(f"Inventario: {len(self.cpu_player.inventory)} pedidos")
            print(f"Pedidos activos: {len(self.active_orders)}")
            
            # Verificar TODOS los pedidos activos con sus posiciones accesibles
            for i, order in enumerate(self.active_orders):
                pickup_x, pickup_y = order.pickup
                accessible_position = self.cpu_player._get_nearest_accessible_position(order.pickup, self.game_map)
                
                if accessible_position:
                    distance = max(abs(self.cpu_player.grid_x - accessible_position[0]), 
                                abs(self.cpu_player.grid_y - accessible_position[1]))
                    print(f"Pedido {i}: {order.id}")
                    print(f"  Pickup edificio: {order.pickup}")
                    print(f"  Posición accesible: {accessible_position}")
                    print(f"  Distancia a accesible: {distance}")
                    
                    if distance <= 3:
                        print(f"  -> CERCA! Puede recoger desde aquí")

# Llamar este método periódicamente o con una tecla de debug

if __name__ == "__main__":
    setup_directories()
    
    # Probar con diferentes dificultades:
    # game = GameEngine()  # Sin CPU
    #game = GameEngine(cpu_difficulty="easy")    # CPU fácil
    #game = GameEngine(cpu_difficulty="medium")  # CPU medio
    game = GameEngine(cpu_difficulty="medium")    # CPU difícil
    
    game.run()