#!/usr/bin/env python3
"""
AutoCoder - Autonomous Coding Agent
A terminal-based AI coding assistant powered by DeepSeek R1
"""

import os
import sys
import json
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from api_client import DeepSeekClient
from project_manager import ProjectManager
from executor import CodeExecutor
from logger import Logger
from ui import CLI
from error_handler import ErrorHandler


class AutoCoder:
    """Main application class for the autonomous coding agent"""
    
    def __init__(self):
        self.config = Config()
        self.logger = Logger()
        self.client = DeepSeekClient(self.config.api_key)
        self.project_manager = ProjectManager()
        self.executor = CodeExecutor()
        self.ui = CLI()
        self.error_handler = ErrorHandler(self.client, self.logger)
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.ui.print_info("\nShutting down gracefully...")
        self.running = False
        sys.exit(0)
    
    def initialize(self) -> bool:
        """Initialize the application and check prerequisites"""
        try:
            # Check API key
            if not self.config.api_key:
                self.ui.print_error("OpenRouter API key not found!")
                self.ui.print_info("Please set OPENROUTER_API_KEY environment variable")
                self.ui.print_info("or create config/config.json with your API key")
                return False
            
            # Test API connection
            self.ui.print_info("Testing API connection...")
            if not self.client.test_connection():
                self.ui.print_error("Failed to connect to OpenRouter API")
                return False
            
            self.ui.print_success("API connection successful!")
            
            # Create necessary directories
            os.makedirs("projects", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            os.makedirs("config", exist_ok=True)
            
            return True
            
        except Exception as e:
            self.ui.print_error(f"Initialization failed: {str(e)}")
            return False
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            return
        
        self.ui.show_welcome()
        
        while self.running:
            try:
                command = self.ui.get_command()
                
                if not command or command.strip() == "":
                    continue
                
                self._handle_command(command.strip())
                
            except KeyboardInterrupt:
                self.ui.print_info("\nUse 'exit' to quit gracefully")
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                self.ui.print_error(f"An unexpected error occurred: {str(e)}")
    
    def _handle_command(self, command: str):
        """Handle user commands"""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in ["exit", "quit", "q"]:
            self.running = False
            self.ui.print_info("Goodbye!")
            
        elif cmd == "new":
            self._handle_new_project(args)
            
        elif cmd == "run":
            self._handle_run_project(args)
            
        elif cmd == "edit":
            self._handle_edit_project(args)
            
        elif cmd == "retry":
            self._handle_retry()
            
        elif cmd == "logs":
            self._handle_show_logs(args)
            
        elif cmd == "list":
            self._handle_list_projects()
            
        elif cmd == "load":
            self._handle_load_project(args)
            
        elif cmd == "status":
            self._handle_status()
            
        elif cmd in ["help", "?"]:
            self.ui.show_help()
            
        else:
            # Treat unknown commands as natural language instructions
            self._handle_natural_language(command)
    
    def _handle_new_project(self, description: str):
        """Create a new project from description"""
        if not description:
            description = self.ui.prompt("Describe your project: ")
        
        if not description:
            self.ui.print_error("Project description is required")
            return
        
        self.ui.print_info(f"Creating new project: {description}")
        
        try:
            # Generate project structure and code
            with self.ui.progress("Generating project..."):
                project_data = self.client.generate_project(description)
            
            if not project_data:
                self.ui.print_error("Failed to generate project")
                return
            
            # Create project files
            with self.ui.progress("Setting up files..."):
                project_path = self.project_manager.create_project(
                    project_data["name"], 
                    project_data
                )
            
            # Install dependencies
            if project_data.get("dependencies"):
                with self.ui.progress("Installing dependencies..."):
                    success = self.executor.install_dependencies(
                        project_path, 
                        project_data["dependencies"]
                    )
                    if not success:
                        self.ui.print_warning("Some dependencies failed to install")
            
            self.project_manager.set_current_project(project_path)
            self.ui.print_success(f"Project created successfully at: {project_path}")
            
            # Auto-run if it's a runnable project
            if project_data.get("auto_run", False):
                self.ui.print_info("Starting project...")
                self._run_current_project()
            
        except Exception as e:
            self.logger.error(f"Failed to create project: {str(e)}")
            self.ui.print_error(f"Project creation failed: {str(e)}")
    
    def _handle_run_project(self, project_name: str = ""):
        """Run the current or specified project"""
        if project_name:
            project_path = self.project_manager.get_project_path(project_name)
            if not project_path:
                self.ui.print_error(f"Project '{project_name}' not found")
                return
            self.project_manager.set_current_project(project_path)
        
        self._run_current_project()
    
    def _run_current_project(self):
        """Run the currently loaded project"""
        current_project = self.project_manager.get_current_project()
        if not current_project:
            self.ui.print_error("No project loaded. Use 'new' to create one or 'load' to load existing.")
            return
        
        try:
            with self.ui.progress("Starting project..."):
                result = self.executor.run_project(current_project)
            
            if result.success:
                self.ui.print_success("Project started successfully!")
                if result.url:
                    self.ui.print_info(f"Server running at: {result.url}")
                if result.output:
                    self.ui.print_output(result.output)
            else:
                self.ui.print_error("Project failed to start")
                if result.error:
                    self.ui.print_error(result.error)
                
                # Auto-retry on failure
                if self.ui.confirm("Would you like me to try to fix the error?"):
                    self._handle_retry()
                    
        except Exception as e:
            self.logger.error(f"Failed to run project: {str(e)}")
            self.ui.print_error(f"Execution failed: {str(e)}")
    
    def _handle_edit_project(self, instruction: str):
        """Edit the current project based on instruction"""
        current_project = self.project_manager.get_current_project()
        if not current_project:
            self.ui.print_error("No project loaded")
            return
        
        if not instruction:
            instruction = self.ui.prompt("What would you like to change? ")
        
        if not instruction:
            self.ui.print_error("Edit instruction is required")
            return
        
        try:
            self.ui.print_info(f"Applying changes: {instruction}")
            
            with self.ui.progress("Analyzing project..."):
                project_info = self.project_manager.get_project_info(current_project)
            
            with self.ui.progress("Generating changes..."):
                changes = self.client.edit_project(project_info, instruction)
            
            if changes:
                with self.ui.progress("Applying changes..."):
                    self.project_manager.apply_changes(current_project, changes)
                
                self.ui.print_success("Changes applied successfully!")
                
                # Ask if user wants to run the updated project
                if self.ui.confirm("Would you like to run the updated project?"):
                    self._run_current_project()
            else:
                self.ui.print_error("Failed to generate changes")
                
        except Exception as e:
            self.logger.error(f"Failed to edit project: {str(e)}")
            self.ui.print_error(f"Edit failed: {str(e)}")
    
    def _handle_retry(self):
        """Retry the last failed operation with error fixing"""
        current_project = self.project_manager.get_current_project()
        if not current_project:
            self.ui.print_error("No project loaded")
            return
        
        try:
            self.ui.print_info("Analyzing and fixing errors...")
            
            with self.ui.progress("Diagnosing issues..."):
                success = self.error_handler.fix_project_errors(current_project)
            
            if success:
                self.ui.print_success("Errors fixed! Trying to run again...")
                self._run_current_project()
            else:
                self.ui.print_error("Could not automatically fix the errors")
                
        except Exception as e:
            self.logger.error(f"Retry failed: {str(e)}")
            self.ui.print_error(f"Retry failed: {str(e)}")
    
    def _handle_show_logs(self, filter_term: str = ""):
        """Show application logs"""
        logs = self.logger.get_recent_logs(50)
        
        if filter_term:
            logs = [log for log in logs if filter_term.lower() in log.lower()]
        
        if logs:
            self.ui.print_info("Recent logs:")
            for log in logs:
                print(log)
        else:
            self.ui.print_info("No logs found")
    
    def _handle_list_projects(self):
        """List all available projects"""
        projects = self.project_manager.list_projects()
        current = self.project_manager.get_current_project()
        
        if projects:
            self.ui.print_info("Available projects:")
            for project in projects:
                marker = " (current)" if project == current else ""
                self.ui.print_info(f"  - {project}{marker}")
        else:
            self.ui.print_info("No projects found")
    
    def _handle_load_project(self, project_name: str):
        """Load an existing project"""
        if not project_name:
            projects = self.project_manager.list_projects()
            if not projects:
                self.ui.print_error("No projects available")
                return
            
            self.ui.print_info("Available projects:")
            for i, project in enumerate(projects, 1):
                self.ui.print_info(f"  {i}. {project}")
            
            choice = self.ui.prompt("Select project (number or name): ")
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(projects):
                    project_name = projects[idx]
                else:
                    self.ui.print_error("Invalid selection")
                    return
            else:
                project_name = choice
        
        project_path = self.project_manager.get_project_path(project_name)
        if project_path:
            self.project_manager.set_current_project(project_path)
            self.ui.print_success(f"Loaded project: {project_name}")
        else:
            self.ui.print_error(f"Project '{project_name}' not found")
    
    def _handle_status(self):
        """Show current status"""
        current_project = self.project_manager.get_current_project()
        
        self.ui.print_info("=== AutoCoder Status ===")
        self.ui.print_info(f"API Status: Connected")
        self.ui.print_info(f"Current Project: {current_project or 'None'}")
        
        if current_project:
            project_info = self.project_manager.get_project_info(current_project)
            self.ui.print_info(f"Project Type: {project_info.get('type', 'Unknown')}")
            self.ui.print_info(f"Language: {project_info.get('language', 'Unknown')}")
    
    def _handle_natural_language(self, instruction: str):
        """Handle natural language instructions"""
        current_project = self.project_manager.get_current_project()
        
        if current_project:
            # If there's a current project, treat as edit instruction
            self._handle_edit_project(instruction)
        else:
            # If no current project, treat as new project creation
            self._handle_new_project(instruction)


def main():
    """Entry point for the application"""
    parser = argparse.ArgumentParser(description="AutoCoder - Autonomous Coding Agent")
    parser.add_argument("--version", action="version", version="AutoCoder 1.0.0")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("command", nargs="*", help="Command to execute")
    
    args = parser.parse_args()
    
    try:
        app = AutoCoder()
        
        # If command line arguments provided, execute and exit
        if args.command:
            if not app.initialize():
                sys.exit(1)
            
            command = " ".join(args.command)
            app._handle_command(command)
        else:
            # Run interactive mode
            app.run()
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()