import pygame
import sys
import time
from ui.main_menu import MainMenu
from utils.setup_directories import setup_directories
from utils.score_manager import initialize_score_system

def show_loading_screen(screen, difficulty):
    """Muestra una pantalla de carga con animación"""
    start_time = time.time()
    loading = True
    
    # Configurar fuentes
    try:
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 24)
    except:
        font_large = pygame.font.SysFont("Arial", 48)
        font_medium = pygame.font.SysFont("Arial", 32)
        font_small = pygame.font.SysFont("Arial", 24)
    
    while loading:
        current_time = time.time() - start_time
        
        # Manejar eventos (permitir salir durante la carga)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False  # Cancelar carga
        
        # Dibujar fondo
        screen.fill((30, 30, 60))
        
        # Título
        title = font_large.render("COURIER QUEST", True, (255, 215, 0))
        screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, 150))
        
        # Mensaje de carga
        if difficulty:
            diff_name = {
                "easy": "FÁCIL",
                "medium": "MEDIO", 
                "hard": "DIFÍCIL",
                "none": "SIN CPU"
            }.get(difficulty, difficulty.upper())
            
            loading_text = font_medium.render(f"Iniciando partida - {diff_name}", True, (200, 200, 200))
        else:
            loading_text = font_medium.render("Cargando partida guardada...", True, (200, 200, 200))
        
        screen.blit(loading_text, (screen.get_width() // 2 - loading_text.get_width() // 2, 250))
        
        # Animación de puntos suspensivos
        dots = "." * (int(current_time * 2) % 4)
        progress_text = font_small.render(f"Cargando recursos{dots}", True, (150, 150, 150))
        screen.blit(progress_text, (screen.get_width() // 2 - progress_text.get_width() // 2, 300))
        
        # Barra de progreso simulada
        progress_width = min(400, (current_time * 100))  # Simula progreso
        pygame.draw.rect(screen, (100, 100, 100), (screen.get_width() // 2 - 200, 350, 400, 20))
        pygame.draw.rect(screen, (0, 200, 0), (screen.get_width() // 2 - 200, 350, progress_width, 20))
        
        # Instrucción para cancelar
        cancel_text = font_small.render("Presiona ESC para cancelar", True, (100, 100, 100))
        screen.blit(cancel_text, (screen.get_width() // 2 - cancel_text.get_width() // 2, 400))
        
        pygame.display.flip()
        
        # Simular un tiempo mínimo de carga (opcional)
        if current_time > 1.0:  # Mínimo 1 segundo de pantalla de carga
            loading = False
            
        pygame.time.delay(50)
    
    return True

def main():
    pygame.init()
    
    setup_directories()
    
    score_success = initialize_score_system()
    if not score_success:
        print("Continuando sin sistema de puntuación...")
    
    while True:
        try:
            screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Courier Quest")
            menu = MainMenu(screen)
            clock = pygame.time.Clock()
            
            menu_running = True
            load_slot = None
            cpu_difficulty = None
            
            while menu_running:
                action = menu.handle_events()
                
                if action == "quit":
                    pygame.quit()
                    sys.exit()
                elif action == "start_game":
                    menu_running = False
                    cpu_difficulty = menu.cpu_difficulty
                    print(f"Iniciando juego con dificultad CPU: {cpu_difficulty}")
                    
                    # MOSTRAR PANTALLA DE CARGA
                    continue_loading = show_loading_screen(screen, cpu_difficulty)
                    if not continue_loading:
                        continue  # Volver al menú si se canceló
                    
                elif action and action.startswith("load_"):
                    load_slot = action[5:]  
                    cpu_difficulty = None
                    menu_running = False
                    
                    # MOSTRAR PANTALLA DE CARGA
                    continue_loading = show_loading_screen(screen, None)
                    if not continue_loading:
                        continue  # Volver al menú si se canceló
                
                menu.draw()
                pygame.display.flip()
                clock.tick(60)
            
            from game_engine import GameEngine
            # Pasar la dificultad al GameEngine
            game = GameEngine(load_slot=load_slot, cpu_difficulty=cpu_difficulty)
            game.run()
            
        except pygame.error as e:
            if "display Surface quit" in str(e):
                continue
            else:
                print(f"Error de Pygame: {e}")
                break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    pygame.quit()
    sys.exit()  

if __name__ == "__main__":
    main()