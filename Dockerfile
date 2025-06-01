# === Stage 1: Maven build with Java 21 ===
FROM registry.access.redhat.com/ubi8/openjdk-21:1.18 as build

# Workdir inside the container
WORKDIR /app

# Copy the Maven project files
COPY . .

# Build the app
RUN mvn clean package -DskipTests

# === Stage 2: Minimal runtime image ===
FROM registry.access.redhat.com/ubi8/openjdk-21-runtime:1.18

WORKDIR /app

# Copy only the built JAR
COPY --from=build /app/target/*.jar app.jar

# Run the app
CMD ["java", "-jar", "app.jar"]
