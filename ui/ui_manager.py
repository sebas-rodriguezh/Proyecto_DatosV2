import pygame
from datetime import datetime

class UIManager:
    """Gestor de la interfaz de usuario del juego"""
    
    def __init__(self, screen, game_map, screen_width, screen_height):
        self.screen = screen
        self.game_map = game_map
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Configurar fuentes
        self.setup_fonts()
        
        # Colores para los trabajos
        self.job_colors = [
            (255, 100, 100), (100, 100, 255), (255, 255, 100),
            (255, 100, 255), (100, 255, 255)
        ]
        
        # Variables para mensajes
        self.message = ""
        self.message_timer = 0
        self.selected_order = None
        
        self.show_inventory_controls = False
        self.controls_timer = 0
        self.controls_duration = 5.0  # Segundos que se muestran los controles
    
    def setup_fonts(self):
        """Configura las fuentes del juego"""
        try:
            self.font_small = pygame.font.Font(None, 12)
            self.font_medium = pygame.font.Font(None, 14)
            self.font_large = pygame.font.Font(None, 18)
            self.font_xlarge = pygame.font.Font(None, 24)
        except:
            self.font_small = pygame.font.SysFont("Arial", 12)
            self.font_medium = pygame.font.SysFont("Arial", 14)
            self.font_large = pygame.font.SysFont("Arial", 18, bold=True)
            self.font_xlarge = pygame.font.SysFont("Arial", 24, bold=True)
    
    def handle_event(self, event, active_orders, player):
        """Maneja eventos relacionados con la UI"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            
            # Verificar clic en el panel lateral
            if mouse_x > self.game_map.width * self.game_map.tile_size:
                self.handle_sidebar_click(mouse_x, mouse_y, active_orders, player)
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p or event.key == pygame.K_e:
                self.show_inventory_controls = True
                self.controls_timer = self.controls_duration
    
    def handle_sidebar_click(self, mouse_x, mouse_y, active_orders, player):
        """Maneja clics en el panel lateral"""
        # Calcular posición de la lista de trabajos
        weather_y_pos = 300 if not player.inventory else 235 + len(player.inventory) * 40 + 10
        jobs_y_pos = weather_y_pos + 85
        order_index = (mouse_y - jobs_y_pos - 25) // 70
        
        if 0 <= order_index < len(active_orders):
            self.selected_order = active_orders[order_index]
            self.show_message(f"Seleccionado: {self.selected_order.id}", 3)
    
    def show_message(self, message, duration):
        """Muestra un mensaje temporal"""
        self.message = message
        self.message_timer = duration
    
    def update_messages(self, dt):
        """Actualiza los mensajes temporales"""
        if self.message_timer > 0:
            self.message_timer -= dt
        else:
            self.message = ""
        

        if self.show_inventory_controls:
            self.controls_timer -= dt
            if self.controls_timer <= 0:
                self.show_inventory_controls = False
    
    def draw_sidebar(self, player, active_orders, weather_system, game_time, game_state, pending_count=0, cpu_player=None):
        """Dibuja el panel lateral completo """
        cols = self.game_map.width
        sidebar_rect = pygame.Rect(cols * self.game_map.tile_size, 0, 300, self.screen_height)
        pygame.draw.rect(self.screen, (240, 240, 240), sidebar_rect)
        pygame.draw.line(self.screen, (200, 200, 200), 
                        (cols * self.game_map.tile_size, 0), 
                        (cols * self.game_map.tile_size, self.screen_height), 2)
        
        x_offset = cols * self.game_map.tile_size
        
        self.draw_header(cols, game_time, game_state)
        self.draw_player_status(cols, player)
        self.draw_inventory(cols, player, game_time)  
        self.draw_weather_info(cols, weather_system, player)
        self.draw_available_jobs(cols, active_orders, weather_system, player, pending_count)
        
        # Controles
        if self.show_inventory_controls:
            self.draw_inventory_controls_popup()
        else:
            self.draw_inventory_controls_hint()
            
        if cpu_player:
            self.draw_cpu_status(cols, cpu_player)

    def draw_cpu_status(self, cols, cpu_player):
        """Dibuja el estado del jugador CPU"""
        x_offset = cols * self.game_map.tile_size
        cpu_y_pos = self.screen_height - 120
        
        # Título del CPU
        cpu_title = self.font_medium.render(f"CPU ({cpu_player.difficulty}):", True, (0, 0, 0))
        self.screen.blit(cpu_title, (x_offset + 10, cpu_y_pos))
        
        # Estadísticas
        stats_text = self.font_small.render(
            f"Pedidos: {cpu_player.orders_completed} | Ganancia: ${cpu_player.total_earnings}", 
            True, (0, 0, 0)
        )
        self.screen.blit(stats_text, (x_offset + 10, cpu_y_pos + 20))
        
        # Estado actual
        status_text = self.font_small.render(
            f"Posición: ({cpu_player.grid_x}, {cpu_player.grid_y})", 
            True, (100, 100, 100)
        )
        self.screen.blit(status_text, (x_offset + 10, cpu_y_pos + 35))
        
        # Barra de stamina del CPU
        stamina_text = self.font_small.render(f"Resistencia: {int(cpu_player.stamina)}/100", True, (0, 0, 0))
        self.screen.blit(stamina_text, (x_offset + 10, cpu_y_pos + 50))
        self.draw_bar(x_offset + 10, cpu_y_pos + 65, 150, 10, cpu_player.stamina / 100, (200, 0, 0))

    
    def draw_inventory_controls_popup(self):
        """Dibuja un popup emergente con los controles de inventario en esquina inferior izquierda"""
        # Posición en esquina inferior izquierda 
        popup_width = 280
        popup_height = 70
        popup_x = 10  # Esquina izquierda
        popup_y = self.screen_height - popup_height - 10  # Esquina inferior
        
        # Fondo del popup con transparencia
        popup_surface = pygame.Surface((popup_width, popup_height), pygame.SRCALPHA)
        popup_surface.fill((240, 240, 240, 230))  # Fondo semi-transparente
        
        # Borde
        pygame.draw.rect(popup_surface, (0, 150, 0, 200), 
                        (0, 0, popup_width, popup_height), 2)
        
        self.screen.blit(popup_surface, (popup_x, popup_y))
        
        # Título
        title = self.font_medium.render("🎮 CONTROLES INVENTARIO", True, (0, 100, 0))
        self.screen.blit(title, (popup_x + 10, popup_y + 5))
        
        controls = [
            "P: Ordenar por Prioridad",
            "O: Ordenar por Urgencia (Deadline)"
        ]
        
        for i, control in enumerate(controls):
            text = self.font_small.render(control, True, (0, 0, 0))
            self.screen.blit(text, (popup_x + 20, popup_y + 25 + i * 15))
    
    def draw_inventory_controls_hint(self):
        """Dibuja un pequeño recordatorio de los controles en esquina inferior izquierda"""
        hint_text = self.font_small.render("Presiona P o E para controles de inventario", 
                                         True, (100, 100, 100))
        self.screen.blit(hint_text, (10, self.screen_height - 20))
    
    def draw_header(self, cols, game_time, game_state):
        """Dibuja el encabezado con título, tiempo y ganancias"""
        x_offset = cols * self.game_map.tile_size
        
        # Título
        title = self.font_large.render("Courier Quest", True, (0, 0, 0))
        self.screen.blit(title, (x_offset + 10, 10))
        
        game_time_text = self.font_medium.render(f"HORA ACTUAL: {game_time.get_game_time_formatted()}", True, (0, 100, 150))
        time_text_width = game_time_text.get_width()
        self.screen.blit(game_time_text, (x_offset + 280 - time_text_width, 12))
        
        # El resto del código EXACTAMENTE IGUAL
        time_bg = pygame.Rect(x_offset + 10, 35, 280, 25)
        pygame.draw.rect(self.screen, (220, 220, 220), time_bg, border_radius=5)
        pygame.draw.rect(self.screen, (100, 100, 100), time_bg, 2, border_radius=5)
        
        # Color del tiempo según urgencia
        remaining_time = game_time.get_remaining_real_time()
        if remaining_time < 60:
            time_color = (255, 50, 50)
        elif remaining_time < 300:
            time_color = (255, 150, 50)
        else:
            time_color = (0, 100, 0)
        
        time_text = self.font_medium.render(f" Tiempo: {game_time.get_remaining_time_formatted()}", True, time_color)
        self.screen.blit(time_text, (x_offset + 20, 38))
        
        # Ganancias y meta
        earnings_text = self.font_medium.render(f" Ganancias: ${game_state.total_earnings}", True, (0, 100, 0))
        self.screen.blit(earnings_text, (x_offset + 10, 65))
        
        goal_text = self.font_small.render(f" Meta: ${game_state.income_goal}", True, (0, 0, 0))
        self.screen.blit(goal_text, (x_offset + 150, 65))

    def draw_player_status(self, cols, player):
        """Dibuja el estado del jugador"""
        x_offset = cols * self.game_map.tile_size
        
        # Título
        player_title = self.font_medium.render("Estado del Repartidor:", True, (0, 0, 0))
        self.screen.blit(player_title, (x_offset + 10, 90))
        
        # Barra de resistencia
        stamina_text = self.font_small.render(f"Resistencia: {int(player.stamina)}/100", 
                                            True, (0, 0, 0))
        self.screen.blit(stamina_text, (x_offset + 10, 115))
        self.draw_bar(x_offset + 10, 130, 150, 15, player.stamina / 100, (0, 200, 0))
        
        # Reputación
        reputation_text = self.font_small.render(f"Reputación: {player.reputation}/100", 
                                               True, (0, 0, 0))
        self.screen.blit(reputation_text, (x_offset + 10, 150))
        
        # Color de la barra de reputación según el nivel
        if player.reputation >= 90:
            rep_color = (0, 200, 0)
        elif player.reputation >= 70:
            rep_color = (0, 150, 200)
        elif player.reputation >= 50:
            rep_color = (255, 150, 0)
        else:
            rep_color = (255, 50, 50)
        
        self.draw_bar(x_offset + 10, 165, 150, 15, player.reputation / 100, rep_color)
        
        # Peso actual
        weight_text = self.font_small.render(f"Peso: {player.current_weight}/{player.max_weight}", 
                                           True, (0, 0, 0))
        self.screen.blit(weight_text, (x_offset + 10, 185))
    
    def draw_bar(self, x, y, width, height, progress, color):
        """Dibuja una barra de progreso"""
        pygame.draw.rect(self.screen, (200, 200, 200), (x, y, width, height))
        pygame.draw.rect(self.screen, color, (x, y, width * progress, height))
    

    def draw_inventory(self, cols, player, game_time=None):
        """Dibuja el inventario mostrando colores originales y tiempo restante"""
        x_offset = cols * self.game_map.tile_size
        
        inventory_title = self.font_medium.render("Inventario (Click derecho sobre el pedido para cancelar):", True, (0, 0, 0))
        self.screen.blit(inventory_title, (x_offset + 10, 210))
        
        if player.inventory:
            for i, order in enumerate(player.inventory):
                y_pos = 235 + i * 45
                

                bg_color = order.color
                
                light_bg_color = (
                    min(255, bg_color[0] + 50),
                    min(255, bg_color[1] + 50), 
                    min(255, bg_color[2] + 50)
                )
                
                if game_time:
                    current_time = game_time.get_current_game_time()
                    time_remaining = order.get_time_remaining(current_time)
                    minutes = int(time_remaining // 60)
                    seconds = int(time_remaining % 60)
                    time_text = f"{minutes:02d}:{seconds:02d}"
                else:
                    time_text = order.deadline.strftime('%H:%M')
                
                # Determinar color de borde basado en prioridad
                if order.priority > 0:
                    border_color = (200, 0, 0)   # Rojo para prioritarios
                    priority_icon = "🚨 "  # Icono de alerta
                else:
                    border_color = (0, 150, 0)   # Verde para normales
                    priority_icon = ""           # Sin icono
                
                # Dibujar caja de inventario
                pygame.draw.rect(self.screen, light_bg_color, (x_offset + 10, y_pos, 280, 40))
                pygame.draw.rect(self.screen, border_color, (x_offset + 10, y_pos, 280, 40), 2)
                
                # Información del pedido CON TIEMPO RESTANTE
                order_text = f"{priority_icon}{order.id} | {time_text}"
                order_surface = self.font_small.render(order_text, True, (0, 0, 0))
                self.screen.blit(order_surface, (x_offset + 15, y_pos + 5))
                
                # Información adicional
                info_text = f"P:{order.priority} | ${order.payout} | {order.weight}kg"
                info_surface = self.font_small.render(info_text, True, (80, 80, 80))
                self.screen.blit(info_surface, (x_offset + 15, y_pos + 20))
        else:
            no_items = self.font_small.render("No hay pedidos en inventario", True, (150, 150, 150))
            self.screen.blit(no_items, (x_offset + 15, 235))

    def draw_weather_info(self, cols, weather_system, player):
        """Dibuja información del clima"""
        x_offset = cols * self.game_map.tile_size
        
        weather_title = self.font_medium.render("Condición Climática:", True, (0, 0, 0))
        weather_y_pos = 300 if not player.inventory else 235 + len(player.inventory) * 40 + 10
        self.screen.blit(weather_title, (x_offset + 10, weather_y_pos))
        
        # Dibujar indicador del clima
        weather_system.draw(self.screen, x_offset + 40, weather_y_pos + 35)
        
        # Información textual del clima
        condition_name = weather_system.current_condition.value.replace("_", " ").title()
        weather_text = self.font_small.render(condition_name, True, (0, 0, 0))
        self.screen.blit(weather_text, (x_offset + 60, weather_y_pos + 25))
        
        intensity_text = self.font_small.render(f"Intensidad: {weather_system.current_intensity:.2f}", 
                                              True, (0, 0, 0))
        self.screen.blit(intensity_text, (x_offset + 60, weather_y_pos + 40))
        
        speed_text = self.font_small.render(f"Velocidad: x{weather_system.current_multiplier:.2f}", 
                                          True, (0, 0, 0))
        self.screen.blit(speed_text, (x_offset + 60, weather_y_pos + 55))
    
    def draw_available_jobs(self, cols, active_orders, weather_system, player, pending_count=0):
        """Dibuja la lista de trabajos disponibles (SOLO los activos)"""
        x_offset = cols * self.game_map.tile_size
        
        weather_y_pos = 300 if not player.inventory else 235 + len(player.inventory) * 40 + 10
        jobs_title = self.font_medium.render("Trabajos Disponibles:", True, (0, 0, 0))
        jobs_y_pos = weather_y_pos + 85
        self.screen.blit(jobs_title, (x_offset + 10, jobs_y_pos))
        
        # Mostrar solo pedidos activos (no pendientes)
        for i, order in enumerate(active_orders):
            y_pos = jobs_y_pos + 25 + i * 70
            
            pygame.draw.rect(self.screen, self.job_colors[i % len(self.job_colors)], 
                        (x_offset + 10, y_pos, 20, 20))
            
            order_id = self.font_small.render(f"ID: {order.id}", True, (0, 0, 0))
            self.screen.blit(order_id, (x_offset + 35, y_pos))
            
            payout = self.font_small.render(f"Pago: ${order.payout}", True, (0, 0, 0))
            self.screen.blit(payout, (x_offset + 35, y_pos + 15))
            
            deadline = self.font_small.render(f"Entrega: {order.deadline.strftime('%H:%M')}", 
                                            True, (0, 0, 0))
            self.screen.blit(deadline, (x_offset + 10, y_pos + 35))
            
            weight = self.font_small.render(f"Peso: {order.weight}", True, (0, 0, 0))
            self.screen.blit(weight, (x_offset + 120, y_pos + 35))
            
            priority = self.font_small.render(f"Prioridad: {order.priority}", True, (0, 0, 0))
            self.screen.blit(priority, (x_offset + 10, y_pos + 50))
        
        # Mostrar contador de pedidos pendientes
        if pending_count > 0:
            pending_text = self.font_small.render(f" {pending_count} pedidos pendientes...", 
                                            True, (100, 100, 100))
            self.screen.blit(pending_text, (x_offset + 10, jobs_y_pos + 25 + len(active_orders) * 70))


    def draw_legend(self, cols):
        """Dibuja la leyenda del mapa"""
        x_offset = cols * self.game_map.tile_size
        
        legend_title = self.font_medium.render("Leyenda del Mapa:", True, (0, 0, 0))
        legend_y_pos = self.screen_height - 150
        self.screen.blit(legend_title, (x_offset + 10, legend_y_pos))
        
        y_pos = legend_y_pos + 20
        for char, info in self.game_map.legend.items():
            color = self.game_map.COLORS.get(char, (100, 100, 255))
            pygame.draw.rect(self.screen, color, (x_offset + 10, y_pos, 15, 15))
            name = self.font_small.render(info["name"].title(), True, (0, 0, 0))
            self.screen.blit(name, (x_offset + 30, y_pos))
            y_pos += 20
    

    def draw_order_markers(self, active_orders, player, camera_x, camera_y):
        """Dibuja los marcadores de trabajos en el mapa - INCLUYE PEDIDOS EN INVENTARIO"""

        for order in active_orders:
            self.draw_single_order_marker(order, player, camera_x, camera_y, False)
        

        for order in player.inventory:
            self.draw_single_order_marker(order, player, camera_x, camera_y, True)

    def draw_single_order_marker(self, order, player, camera_x, camera_y, is_in_inventory):
        """Dibuja los marcadores para una sola orden"""
        # Usar el color del pedido
        color = order.color
        
        # Verificar si el pedido está en el inventario
        in_inventory = is_in_inventory or (player.inventory.find_by_id(order.id) is not None)
        
        # Dibujar punto de recogida (solo si no está en inventario)
        if not in_inventory:
            pickup_x, pickup_y = order.pickup
            pygame.draw.circle(self.screen, color, 
                            (pickup_x * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_x, 
                            pickup_y * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_y), 
                            7)
            pygame.draw.circle(self.screen, (255, 255, 255), 
                            (pickup_x * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_x, 
                            pickup_y * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_y), 
                            7, 1)
        
        dropoff_x, dropoff_y = order.dropoff
        if in_inventory:
            
            pygame.draw.rect(self.screen, color, 
                        (dropoff_x * self.game_map.tile_size + self.game_map.tile_size // 2 - 5 - camera_x, 
                        dropoff_y * self.game_map.tile_size + self.game_map.tile_size // 2 - 5 - camera_y, 
                        10, 10))
            pygame.draw.rect(self.screen, (255, 255, 255), 
                        (dropoff_x * self.game_map.tile_size + self.game_map.tile_size // 2 - 5 - camera_x, 
                        dropoff_y * self.game_map.tile_size + self.game_map.tile_size // 2 - 5 - camera_y, 
                        10, 10), 2)  # Borde más grueso
        else:
            # Normal si no está en inventario
            pygame.draw.rect(self.screen, color, 
                        (dropoff_x * self.game_map.tile_size + self.game_map.tile_size // 2 - 5 - camera_x, 
                        dropoff_y * self.game_map.tile_size + self.game_map.tile_size // 2 - 5 - camera_y, 
                        10, 10))
            pygame.draw.rect(self.screen, (255, 255, 255), 
                        (dropoff_x * self.game_map.tile_size + self.game_map.tile_size // 2 - 5 - camera_x, 
                        dropoff_y * self.game_map.tile_size + self.game_map.tile_size // 2 - 5 - camera_y, 
                        10, 10), 1)
        

        if not in_inventory:
            pickup_x, pickup_y = order.pickup
            pygame.draw.line(self.screen, color, 
                        (pickup_x * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_x, 
                        pickup_y * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_y),
                        (dropoff_x * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_x, 
                        dropoff_y * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_y), 
                        2)

    def draw_messages(self):
        """Dibuja mensajes temporales"""
        if self.message:
            msg_surface = self.font_medium.render(self.message, True, (0, 0, 0))
            self.screen.blit(msg_surface, (10, 10))
    
    def get_interaction_hint(self, game_map):
        """Obtiene pista de interacción"""
        if hasattr(self, 'interaction_manager'):
            return self.interaction_manager.get_interaction_hint(game_map)
        return ""

    # ui_manager.py - CORREGIR método draw_interaction_hints
    def draw_interaction_hints(self, player, active_orders, camera_x, camera_y, game_map=None):
        """Dibuja pistas de interacción cerca del jugador"""
        if hasattr(self, 'interaction_manager') and hasattr(self.interaction_manager, 'game_time'):
            game_time = self.interaction_manager.game_time
        else:
            # Fallback si no hay game_time disponible
            return
        
        interactable_orders = player.get_interactable_orders(active_orders, game_map, radius=20, game_time=game_time)
        
        if interactable_orders:
            # Mostrar pista para la primera orden interactuable
            interaction = interactable_orders[0]
            order_info = interaction['order']
            action = interaction['action']
            
            if action == 'pickup':
                hint_text = f"Recoger {order_info.id}"
            else:  # dropoff
                hint_text = f"Entregar {order_info.id}"
            
            # Usar la posición del punto de interacción
            loc_x, loc_y = interaction['location']
            

            hint_text_surface = self.font_medium.render(hint_text, True, (0, 0, 0))  # Texto negro
            text_width = hint_text_surface.get_width()
            text_height = hint_text_surface.get_height()
            
            # Tamaño compacto
            padding = 6
            hint_bg = pygame.Rect(
                loc_x * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_x - text_width // 2 - padding,
                loc_y * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_y - 30,
                text_width + padding * 2,
                text_height + padding * 2
            )
            
            hint_color = order_info.color
            light_bg_color = (
                min(255, hint_color[0] + 80),
                min(255, hint_color[1] + 80),
                min(255, hint_color[2] + 80)
            )
            
            # Dibujar fondo con color del pedido
            pygame.draw.rect(self.screen, light_bg_color, hint_bg, border_radius=5)
            # Borde del color original
            pygame.draw.rect(self.screen, hint_color, hint_bg, 2, border_radius=5)
            
            # Dibujar texto centrado
            self.screen.blit(
                hint_text_surface,
                (loc_x * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_x - text_width // 2,
                loc_y * self.game_map.tile_size + self.game_map.tile_size // 2 - camera_y - 30 + padding)
            )
    
    def draw_game_over_screen(self, game_state):
        """Dibuja la pantalla de fin de juego"""
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        if game_state.victory:
            result_text = self.font_xlarge.render("¡VICTORIA!", True, (0, 255, 0))
            reason_text = self.font_large.render(f"Has ganado ${game_state.total_earnings}", True, (255, 255, 255))
        else:
            result_text = self.font_xlarge.render("DERROTA", True, (255, 0, 0))
            reason_text = self.font_large.render(game_state.game_over_reason, True, (255, 255, 255))
        
        self.screen.blit(result_text, (self.screen_width // 2 - result_text.get_width() // 2, self.screen_height // 2 - 50))
        self.screen.blit(reason_text, (self.screen_width // 2 - reason_text.get_width() // 2, self.screen_height // 2))
        
        # Mostrar estadísticas finales
        stats = game_state.get_game_stats()
        final_score = self.font_medium.render(f"Puntaje Final: {stats['final_score']}", True, (255, 255, 255))
        self.screen.blit(final_score, (self.screen_width // 2 - final_score.get_width() // 2, self.screen_height // 2 + 25))
        
        orders_text = self.font_small.render(f"Pedidos completados: {stats['orders_completed']}", True, (255, 255, 255))
        self.screen.blit(orders_text, (self.screen_width // 2 - orders_text.get_width() // 2, self.screen_height // 2 + 45))
        
        restart_text = self.font_medium.render("Juego terminado", True, (255, 255, 255))
        self.screen.blit(restart_text, (self.screen_width // 2 - restart_text.get_width() // 2, self.screen_height // 2 + 70))
    
    def update(self, dt):
        """Actualiza el estado del UI Manager"""
        self.update_messages(dt)
    
    def draw_progress_bar(self, x, y, width, height, progress, bg_color=(200, 200, 200), fg_color=(0, 200, 0)):
        """Dibuja una barra de progreso genérica"""
        # Fondo
        pygame.draw.rect(self.screen, bg_color, (x, y, width, height))
        # Progreso
        progress_width = int(width * min(1.0, max(0.0, progress)))
        pygame.draw.rect(self.screen, fg_color, (x, y, progress_width, height))
        # Borde
        pygame.draw.rect(self.screen, (0, 0, 0), (x, y, width, height), 1)
    
    def draw_minimap(self, x, y, size, player, active_orders, game_map):
        """Dibuja un minimapa del área"""
        # Fondo del minimapa
        pygame.draw.rect(self.screen, (50, 50, 50), (x, y, size, size))
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, size, size), 2)
        
        # Escala del minimapa
        scale_x = size / game_map.width
        scale_y = size / game_map.height
        
        # Dibujar elementos del mapa simplificados
        for row_idx, row in enumerate(game_map.tiles):
            for col_idx, tile in enumerate(row):
                tile_x = x + col_idx * scale_x
                tile_y = y + row_idx * scale_y
                
                if tile == "B":  # Edificios
                    pygame.draw.rect(self.screen, (100, 100, 100), 
                                   (tile_x, tile_y, max(1, scale_x), max(1, scale_y)))
                elif tile == "P":  # Parques
                    pygame.draw.rect(self.screen, (0, 100, 0), 
                                   (tile_x, tile_y, max(1, scale_x), max(1, scale_y)))
        
        # Dibujar pedidos activos
        for i, order in enumerate(active_orders):
            color = self.job_colors[i % len(self.job_colors)]
            
            # Punto de recogida
            pickup_x = x + order.pickup[0] * scale_x
            pickup_y = y + order.pickup[1] * scale_y
            pygame.draw.circle(self.screen, color, (int(pickup_x), int(pickup_y)), 2)
            
            # Punto de entrega
            dropoff_x = x + order.dropoff[0] * scale_x
            dropoff_y = y + order.dropoff[1] * scale_y
            pygame.draw.rect(self.screen, color, 
                           (int(dropoff_x-1), int(dropoff_y-1), 2, 2))
        
        # Dibujar jugador
        player_x = x + player.visual_x * scale_x
        player_y = y + player.visual_y * scale_y
        pygame.draw.circle(self.screen, (255, 0, 0), (int(player_x), int(player_y)), 3)
    
    def draw_order_details_popup(self, order, mouse_pos):
        """Dibuja un popup con detalles de un pedido"""
        if not order:
            return
        
        # Contenido del popup
        lines = [
            f"ID: {order.id}",
            f"Recogida: {order.pickup}",
            f"Entrega: {order.dropoff}",
            f"Pago: ${order.payout}",
            f"Peso: {order.weight}kg",
            f"Prioridad: {order.priority}",
            f"Límite: {order.deadline.strftime('%H:%M')}"
        ]
        
        # Calcular tamaño del popup
        max_width = max(self.font_small.get_size(line)[0] for line in lines) + 20
        popup_height = len(lines) * 15 + 10
        
        # Posición del popup (evitar bordes de pantalla)
        popup_x = min(mouse_pos[0] + 10, self.screen_width - max_width - 10)
        popup_y = min(mouse_pos[1] + 10, self.screen_height - popup_height - 10)
        
        # Fondo del popup
        pygame.draw.rect(self.screen, (255, 255, 255), 
                        (popup_x, popup_y, max_width, popup_height))
        pygame.draw.rect(self.screen, (0, 0, 0), 
                        (popup_x, popup_y, max_width, popup_height), 2)
        
        # Dibujar texto
        for i, line in enumerate(lines):
            text_surface = self.font_small.render(line, True, (0, 0, 0))
            self.screen.blit(text_surface, (popup_x + 10, popup_y + 5 + i * 15))
    
    def draw_keyboard_shortcuts(self):
        """Dibuja los atajos de teclado disponibles"""
        shortcuts = [
            "WASD/Flechas: Moverse",
            "E: Interactuar",
            "R: Reiniciar (fin de juego)",
            "Ctrl+S: Guardar",
            "Ctrl+L: Cargar",
            "Ctrl+Z: Deshacer"
        ]
        
        y_start = self.screen_height - len(shortcuts) * 15 - 10
        
        for i, shortcut in enumerate(shortcuts):
            text = self.font_small.render(shortcut, True, (100, 100, 100))
            self.screen.blit(text, (10, y_start + i * 15))