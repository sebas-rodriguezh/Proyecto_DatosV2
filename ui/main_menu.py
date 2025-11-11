import pygame
import sys
from utils.save_load_manager import SaveLoadManager

class MainMenu:
    def __init__(self, screen):
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.show_high_scores = False
        self.show_controls = False
        self.show_difficulty_selection = False
        self.selected_difficulty = 1  # Medio por defecto
        self.cpu_difficulty = None
        self.high_scores = self.load_high_scores()
        
        # Configurar fuentes
        try:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
        except:
            self.font_large = pygame.font.SysFont("Arial", 48)
            self.font_medium = pygame.font.SysFont("Arial", 32)
            self.font_small = pygame.font.SysFont("Arial", 24)
        
        # Opciones del menú
        self.options = [
            {"text": "Nueva Partida", "action": "new_game"},
            {"text": "Cargar Partida", "action": "load_game"},
            {"text": "Ver Puntuaciones", "action": "high_scores"},
            {"text": "Ver Controles", "action": "controls"},
            {"text": "Salir", "action": "quit"}
        ]
        
        # Sistema de guardado/carga
        self.save_manager = SaveLoadManager()
        self.save_slots = self.get_save_slots()
        self.selected_option = 0
        self.selected_save_slot = None
        self.show_save_slots = False
        self.background = self.create_background()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            
            if event.type == pygame.KEYDOWN:
                if self.show_difficulty_selection:
                    if event.key == pygame.K_UP:
                        self.selected_difficulty = (self.selected_difficulty - 1) % 4
                    elif event.key == pygame.K_DOWN:
                        self.selected_difficulty = (self.selected_difficulty + 1) % 4
                    elif event.key == pygame.K_RETURN:
                        # Asignar la dificultad seleccionada
                        difficulties = ["easy", "medium", "hard", None]
                        self.cpu_difficulty = difficulties[self.selected_difficulty]
                        self.show_difficulty_selection = False
                        return "start_game"
                    elif event.key == pygame.K_ESCAPE:
                        self.show_difficulty_selection = False
                        # No retornar nada, solo salir de la selección de dificultad
                        
                elif self.show_save_slots:
                    if event.key == pygame.K_UP:
                        self.selected_save_slot = (self.selected_save_slot - 1) % len(self.save_slots)
                    elif event.key == pygame.K_DOWN:
                        self.selected_save_slot = (self.selected_save_slot + 1) % len(self.save_slots)
                    elif event.key == pygame.K_RETURN:
                        if self.selected_save_slot is not None:
                            slot_name = self.save_slots[self.selected_save_slot]["name"]
                            return f"load_{slot_name}"
                    elif event.key == pygame.K_ESCAPE:
                        self.show_save_slots = False
                        self.selected_save_slot = None
                        
                elif self.show_high_scores:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                        self.show_high_scores = False
                        
                elif self.show_controls:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                        self.show_controls = False
                        
                else:
                    # Menú principal
                    if event.key == pygame.K_UP:
                        self.selected_option = (self.selected_option - 1) % len(self.options)
                    elif event.key == pygame.K_DOWN:
                        self.selected_option = (self.selected_option + 1) % len(self.options)
                    elif event.key == pygame.K_RETURN:
                        action = self.options[self.selected_option]["action"]
                        if action == "load_game":
                            self.show_save_slots = True
                            self.selected_save_slot = 0
                        elif action == "high_scores":
                            self.show_high_scores = True
                        elif action == "controls":
                            self.show_controls = True
                        elif action == "new_game":
                            self.show_difficulty_selection = True
                            self.selected_difficulty = 1
                        else:
                            return action
                    elif event.key == pygame.K_ESCAPE:
                        return "quit"
        
        return None

    def draw(self):
        self.screen.blit(self.background, (0, 0))
        
        if self.show_high_scores:
            self.draw_high_scores()
        elif self.show_controls:
            self.draw_controls()
        elif self.show_save_slots:
            self.draw_save_slots()
        elif self.show_difficulty_selection:
            self.draw_difficulty_selection()
        else:
            self.draw_main_menu()

    def draw_difficulty_selection(self):
        """Dibuja la selección de dificultad del CPU"""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        title = self.font_large.render("SELECCIONAR DIFICULTAD DEL CPU", True, (255, 215, 0))
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 80))
        
        desc = self.font_small.render("Elige el nivel de desafío para competir contra la IA:", True, (200, 200, 200))
        self.screen.blit(desc, (self.width // 2 - desc.get_width() // 2, 130))
        
        difficulties = [
            {"text": "FÁCIL", "value": "easy", "desc": "CPU con comportamiento básico y aleatorio"},
            {"text": "MEDIO", "value": "medium", "desc": "CPU con estrategia greedy mejorada"},
            {"text": "DIFÍCIL", "value": "hard", "desc": "CPU con optimización avanzada (A*)"},
            {"text": "SIN CPU", "value": "none", "desc": "Jugar solo sin oponente"}
        ]
        
        for i, diff in enumerate(difficulties):
            y_pos = 180 + i * 80
            
            if i == self.selected_difficulty:
                color = (255, 215, 0)
                text = self.font_medium.render("> " + diff["text"] + " <", True, color)
                option_rect = pygame.Rect(self.width // 2 - 200, y_pos - 5, 400, 60)
                pygame.draw.rect(self.screen, (50, 50, 80), option_rect, border_radius=8)
                pygame.draw.rect(self.screen, (255, 215, 0), option_rect, 2, border_radius=8)
            else:
                color = (200, 200, 200)
                text = self.font_medium.render(diff["text"], True, color)
            
            self.screen.blit(text, (self.width // 2 - text.get_width() // 2, y_pos))
            
            desc_text = self.font_small.render(diff["desc"], True, (150, 150, 150))
            self.screen.blit(desc_text, (self.width // 2 - desc_text.get_width() // 2, y_pos + 30))
        
        instructions = self.font_small.render("ENTER para confirmar, ESC para volver", True, (150, 150, 150))
        self.screen.blit(instructions, (self.width // 2 - instructions.get_width() // 2, self.height - 50))

    def get_save_slots(self):
        """Obtiene información de partidas guardadas"""
        saves = self.save_manager.list_saves()
        save_slots = []
        
        for i in range(1, 4):
            slot_name = f"slot{i}"
            if slot_name in saves:
                info = saves[slot_name]
                save_slots.append({
                    "name": slot_name,
                    "info": f"${info.get('earnings', 0)} - {info.get('orders_completed', 0)} pedidos"
                })
            else:
                save_slots.append({
                    "name": slot_name,
                    "info": "Vacío"
                })
        
        return save_slots

    def create_background(self):
        """Crea un fondo atractivo para el menú"""
        background = pygame.Surface((self.width, self.height))
        background.fill((30, 30, 60))
        
        for i in range(0, self.width, 20):
            pygame.draw.line(background, (50, 50, 100), (i, 0), (i, self.height), 1)
        for i in range(0, self.height, 20):
            pygame.draw.line(background, (50, 50, 100), (0, i), (self.width, i), 1)
        
        return background

    def draw_main_menu(self):
        """Dibuja el menú principal"""
        title = self.font_large.render("COURIER QUEST", True, (255, 215, 0))
        title_shadow = self.font_large.render("COURIER QUEST", True, (150, 100, 0))
        
        self.screen.blit(title_shadow, (self.width // 2 - title.get_width() // 2 + 2, 102))
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 100))
        
        subtitle = self.font_small.render("Sistema de Entregas", True, (200, 200, 200))
        self.screen.blit(subtitle, (self.width // 2 - subtitle.get_width() // 2, 160))
        
        for i, option in enumerate(self.options):
            if i == self.selected_option:
                color = (255, 215, 0)
                text = self.font_medium.render("> " + option["text"] + " <", True, color)
            else:
                color = (200, 200, 200)
                text = self.font_medium.render(option["text"], True, color)
            
            y_pos = 230 + i * 50
            self.screen.blit(text, (self.width // 2 - text.get_width() // 2, y_pos))
        
        instructions = self.font_small.render("Usa ↑↓ para navegar, ENTER para seleccionar, ESC para salir", 
                                            True, (150, 150, 150))
        self.screen.blit(instructions, (self.width // 2 - instructions.get_width() // 2, self.height - 50))

    def draw_save_slots(self):
        """Dibuja la selección de partidas guardadas"""
        title = self.font_large.render("SELECCIONAR PARTIDA", True, (255, 215, 0))
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 50))
        
        for i, slot in enumerate(self.save_slots):
            if i == self.selected_save_slot:
                color = (255, 215, 0)
                slot_text = self.font_medium.render(f"> {slot['name']}: {slot['info']} <", True, color)
            else:
                color = (200, 200, 200)
                slot_text = self.font_medium.render(f"{slot['name']}: {slot['info']}", True, color)
            
            y_pos = 150 + i * 50
            self.screen.blit(slot_text, (self.width // 2 - slot_text.get_width() // 2, y_pos))
        
        instructions = self.font_small.render("ENTER para cargar, ESC para volver", 
                                            True, (150, 150, 150))
        self.screen.blit(instructions, (self.width // 2 - instructions.get_width() // 2, self.height - 50))

    def draw_controls(self):
        """Dibuja la pantalla de controles"""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        title = self.font_large.render("CONTROLES DEL JUEGO", True, (255, 215, 0))
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 40))
        
        controls = [
            ("MOVIMIENTO", ""),
            ("W/A/S/D o Flechas", "Mover al jugador"),
            ("", ""),
            ("INTERACCION", ""),
            ("E", "Recoger/Entregar pedido"),
            ("Y", "Aceptar pedido en popup"),
            ("N", "Rechazar pedido en popup"),
            ("Clic Derecho", "Cancelar pedido del inventario"),
            ("", ""),
            ("ORGANIZACION", ""),
            ("P", "Ordenar inventario por prioridad"),
            ("O", "Ordenar inventario por urgencia"),
            ("", ""),
            ("SISTEMA", ""),
            ("ESC", "Pausar juego / Cerrar popup"),
            ("Ctrl+S", "Guardar partida (slot 1)"),
            ("Ctrl+Z", "Deshacer última acción"),
            ("Ctrl+Y", "Rehacer acción")
        ]
        
        y_offset = 120
        for control, description in controls:
            if control == "" and description == "":
                y_offset += 10
                continue
            
            if description == "":
                text = self.font_medium.render(control, True, (100, 200, 255))
                self.screen.blit(text, (self.width // 2 - text.get_width() // 2, y_offset))
                y_offset += 35
            else:
                control_text = self.font_small.render(control, True, (255, 215, 0))
                self.screen.blit(control_text, (self.width // 2 - 200, y_offset))
                
                desc_text = self.font_small.render(description, True, (200, 200, 200))
                self.screen.blit(desc_text, (self.width // 2 - 50, y_offset))
                
                y_offset += 28
        
        instructions = self.font_small.render("Presiona ESC o ENTER para volver", True, (150, 150, 150))
        self.screen.blit(instructions, (self.width // 2 - instructions.get_width() // 2, self.height - 50))

    def load_high_scores(self):
        """Carga las puntuaciones desde el archivo"""
        try:
            from utils.score_manager import score_manager
            score_manager.initialize_score_system()
            return score_manager.get_top_scores(10)
        except:
            return []

    def draw_high_scores(self):
        """Dibuja la pantalla de puntuaciones altas"""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        title = self.font_large.render("MEJORES PUNTUACIONES", True, (255, 215, 0))
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 50))
        
        headers = ["POS", "JUGADOR", "PUNTUACIÓN", "GANANCIAS", "FECHA"]
        header_y = 120
        col_widths = [80, 150, 120, 120, 200]
        x_pos = self.width // 2 - sum(col_widths) // 2
        
        for i, header in enumerate(headers):
            text = self.font_small.render(header, True, (255, 215, 0))
            self.screen.blit(text, (x_pos, header_y))
            x_pos += col_widths[i]
        
        if not self.high_scores:
            no_scores = self.font_medium.render("No hay puntuaciones guardadas", True, (200, 200, 200))
            self.screen.blit(no_scores, (self.width // 2 - no_scores.get_width() // 2, 180))
        else:
            for i, score in enumerate(self.high_scores[:10]):
                y_pos = 160 + i * 30
                x_pos = self.width // 2 - sum(col_widths) // 2
                
                pos_text = self.font_small.render(f"{i+1}.", True, (255, 255, 255))
                self.screen.blit(pos_text, (x_pos, y_pos))
                x_pos += col_widths[0]
                
                name_text = self.font_small.render(score.get("player_name", "Jugador"), True, (255, 255, 255))
                self.screen.blit(name_text, (x_pos, y_pos))
                x_pos += col_widths[1]
                
                score_text = self.font_small.render(str(score.get("score", 0)), True, (255, 255, 255))
                self.screen.blit(score_text, (x_pos, y_pos))
                x_pos += col_widths[2]
                
                earnings_text = self.font_small.render(f"${score.get('earnings', 0)}", True, (255, 255, 255))
                self.screen.blit(earnings_text, (x_pos, y_pos))
                x_pos += col_widths[3]
                
                date_str = score.get("date", "")
                if date_str:
                    try:
                        from datetime import datetime
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime("%d/%m/%Y %H:%M")
                    except:
                        formatted_date = date_str[:16]
                else:
                    formatted_date = "N/A"
                
                date_text = self.font_small.render(formatted_date, True, (255, 255, 255))
                self.screen.blit(date_text, (x_pos, y_pos))
        
        instructions = self.font_small.render("Presiona ESC o ENTER para volver", True, (150, 150, 150))
        self.screen.blit(instructions, (self.width // 2 - instructions.get_width() // 2, self.height - 50))