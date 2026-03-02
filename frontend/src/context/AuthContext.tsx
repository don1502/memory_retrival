import { createContext, useContext } from 'react';
import type { ReactNode } from 'react';

interface AuthContextType {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const login = async (email: string, password: string) => {
    void email;
    void password;
    throw new Error('Authentication has been removed');
  };

  const register = async (email: string, password: string, name: string) => {
    void email;
    void password;
    void name;
    throw new Error('Authentication has been removed');
  };

  const logout = () => {
    // no-op
  };

  return (
    <AuthContext.Provider value={{ login, register, logout, isLoading: false }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
