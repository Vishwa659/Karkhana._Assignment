import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.spatial.distance import euclidean

class MobiusStrip:
    def __init__(self, R=1.0, w=0.4, n=200):
        # Basic parameters: radius, width, number of points
        self.R = R
        self.w = w
        self.n = n

        # Create u and v values from parametric equations
        self.u = np.linspace(0, 2 * np.pi, self.n)
        self.v = np.linspace(-self.w / 2, self.w / 2, self.n)
        self.U, self.V = np.meshgrid(self.u, self.v)

        # Compute surface coordinates
        self.X, self.Y, self.Z = self._create_surface()

    def _create_surface(self):
        # Apply Möbius parametric formulas
        X = (self.R + self.V * np.cos(self.U / 2)) * np.cos(self.U)
        Y = (self.R + self.V * np.cos(self.U / 2)) * np.sin(self.U)
        Z = self.V * np.sin(self.U / 2)
        return X, Y, Z

    def compute_surface_area(self):
        # Compute partial derivatives along u and v
        dXu = np.gradient(self.X, axis=1)
        dYu = np.gradient(self.Y, axis=1)
        dZu = np.gradient(self.Z, axis=1)

        dXv = np.gradient(self.X, axis=0)
        dYv = np.gradient(self.Y, axis=0)
        dZv = np.gradient(self.Z, axis=0)

        # Calculate the cross product of the partials
        normal_vectors = np.cross(
            np.stack((dXu, dYu, dZu), axis=2),
            np.stack((dXv, dYv, dZv), axis=2)
        )

        # Surface area = integral of the magnitude of the normals
        area_density = np.linalg.norm(normal_vectors, axis=2)
        area = simpson(simpson(area_density, self.v), self.u)
        return area

    def compute_edge_length(self):
        # Get top and bottom edges of the strip
        edge_top = np.array([self.X[0], self.Y[0], self.Z[0]]).T
        edge_bottom = np.array([self.X[-1], self.Y[-1], self.Z[-1]]).T

        def path_length(edge):
            return sum(
                euclidean(edge[i], edge[i+1]) for i in range(len(edge) - 1)
            ) + euclidean(edge[-1], edge[0])  # close the loop

        return (path_length(edge_top) + path_length(edge_bottom)) / 2

    def plot(self):
        # Display the 3D Möbius surface
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap='coolwarm', edgecolor='k', alpha=0.8)
        ax.set_title("3D Möbius Strip")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        plt.tight_layout()
        plt.show()

# Run the script
if __name__ == "__main__":
    # Input parameters
    R = float(input("Enter radius R (e.g., 1.0): "))
    w = float(input("Enter width w (e.g., 0.4): "))
    n = int(input("Enter resolution n (e.g., 200): "))

    strip = MobiusStrip(R, w, n)
    print(f"Estimated Surface Area: {strip.compute_surface_area():.4f}")
    print(f"Estimated Edge Length: {strip.compute_edge_length():.4f}")
    strip.plot()
