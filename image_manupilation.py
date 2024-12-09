import cv2

def charger_image(chemin):
    image_var = cv2.imread(chemin)

    if image_var is None:
        print(f"Erreur : l'image {chemin} n'a pas pu être chargée. Vérifiez le chemin.")
        exit()
    else:
        return image_var

def afficher_informations(image):
    hauteur = image.shape[0]
    largeur = image.shape[1]
    print(f"Size: {image.size}, Shape: {image.shape}, Largeur: {largeur}, Hauteur: {hauteur}")
    return largeur, hauteur

def clignoter(image, second_image, largeur_image, hauteur_image, afficher_image):
    # Créer une copie de l'image principale
    image_copy = image.copy()

    hauteur_image_2, largeur_image_2 = second_image.shape[:2]

    if afficher_image:
        # Afficher l'image avec l'image d'alerte dans les coins
        image_copy[0:hauteur_image_2, 0:largeur_image_2] = second_image  # Coin supérieur gauche
        image_copy[0:hauteur_image_2, largeur_image-largeur_image_2:largeur_image] = second_image  # Coin supérieur droit
        image_copy[hauteur_image-hauteur_image_2:hauteur_image, 0:largeur_image_2] = second_image  # Coin inférieur gauche
        image_copy[hauteur_image-hauteur_image_2:hauteur_image, largeur_image-largeur_image_2:largeur_image] = second_image  # Coin inférieur droit
    else:
        # Réinitialiser les coins pour créer l'effet de clignotement
        image_copy[0:hauteur_image_2, 0:largeur_image_2] = image[0:hauteur_image_2, 0:largeur_image_2]
        image_copy[0:hauteur_image_2, largeur_image-largeur_image_2:largeur_image] = image[0:hauteur_image_2, largeur_image-largeur_image_2:largeur_image]
        image_copy[hauteur_image-hauteur_image_2:hauteur_image, 0:largeur_image_2] = image[hauteur_image-hauteur_image_2:hauteur_image, 0:largeur_image_2]
        image_copy[hauteur_image-hauteur_image_2:hauteur_image, largeur_image-largeur_image_2:largeur_image] = image[hauteur_image-hauteur_image_2:hauteur_image, largeur_image-largeur_image_2:largeur_image]

    return image_copy

def main():
    image = charger_image("./1.jpg")
    image = cv2.resize(image, (500, 500))

    second_image = charger_image("./alert.png")

    largeur_image, hauteur_image = afficher_informations(image)

    # Placer second_image dans les coins logiques
    new_width = largeur_image // 5
    new_height = hauteur_image // 5

    second_image = cv2.resize(second_image, (new_width, new_height))

    afficher_image = True

    while True:
        # Clignotement de l'image
        image_copy = clignoter(image, second_image, largeur_image, hauteur_image, afficher_image)

        cv2.imshow("Image principale", image_copy)

        key = cv2.waitKey(350) & 0xFF  # Délai de 350ms pour un clignotement

        if key == ord('q') or key == 32:  # 32 est le code ASCII de la barre d'espace
            break

        # Alterner entre l'affichage des coins et la réinitialisation
        afficher_image = not afficher_image

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

"""
Pourquoi le clignotement se désactive si vous mettez un 0 au lieu de 1 dans cv2.waitKey() :

if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    Lorsque vous écrivez cv2.waitKey(0), cela signifie que l'exécution du programme attend indéfiniment jusqu'à ce qu'une touche soit pressée.
    Si vous écrivez cv2.waitKey(1), l'exécution attend une très courte période (1 milliseconde), puis continue immédiatement, ce qui permet d'avoir un "clignotement" ou un rafraîchissement d'affichage presque constant. Ce délai de 1 milliseconde permet de mettre à jour l'affichage plus fréquemment.
    En revanche, avec cv2.waitKey(0), le programme attend que l'utilisateur appuie sur une touche avant de continuer, donc il ne "clignote" pas, car le programme attend indéfiniment et ne poursuit pas l'exécution pour afficher les images répétitivement.
"""