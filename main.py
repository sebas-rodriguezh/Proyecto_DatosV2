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
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False  
        
        
        screen.fill((30, 30, 60))
        
        
        title = font_large.render("COURIER QUEST", True, (255, 215, 0))
        screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, 150))
        
        
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
        
        
        dots = "." * (int(current_time * 2) % 4)
        progress_text = font_small.render(f"Cargando recursos{dots}", True, (150, 150, 150))
        screen.blit(progress_text, (screen.get_width() // 2 - progress_text.get_width() // 2, 300))
        
        
        progress_width = min(400, (current_time * 100))  
        pygame.draw.rect(screen, (100, 100, 100), (screen.get_width() // 2 - 200, 350, 400, 20))
        pygame.draw.rect(screen, (0, 200, 0), (screen.get_width() // 2 - 200, 350, progress_width, 20))
        
        
        cancel_text = font_small.render("Presiona ESC para cancelar", True, (100, 100, 100))
        screen.blit(cancel_text, (screen.get_width() // 2 - cancel_text.get_width() // 2, 400))
        
        pygame.display.flip()
        
        if current_time > 1.0:  
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
                    
                    continue_loading = show_loading_screen(screen, cpu_difficulty)
                    if not continue_loading:
                        continue  
                    
                elif action and action.startswith("load_"):
                    load_slot = action[5:]  
                    cpu_difficulty = None
                    menu_running = False
                    
                    
                    continue_loading = show_loading_screen(screen, None)
                    if not continue_loading:
                        continue  
                
                menu.draw()
                pygame.display.flip()
                clock.tick(60)
            
            from game_engine import GameEngine
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

    #Profe, cuando lo corre, y le da en iniciar nueva partida, al seleccionar el nivel de dificultad, el programa lo "devuelve" al menú principal, ya que en 
    #en ese momento empieza a "dibujar" y cargar el mapa y toda la información. Este pantalla principal se refleja por 2-4 segundos, luego, el juego arranca y funciona normal. 